//! Loopback-only transport for the UI-owned MCP bridge.
//!
//! This module deliberately only moves a single JSON request from a local TCP
//! client to the application and returns the application's JSON reply.  It
//! performs no filesystem, process, or command work itself.

use std::{
    io::{self, BufRead, BufReader, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::{json, Value};

/// The local-only endpoint used by the Python stdio MCP package.
pub const ADDRESS: &str = "127.0.0.1:9756";

const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_CLIENTS: usize = 16;
const QUEUE_CAPACITY: usize = 32;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const IO_TIMEOUT: Duration = Duration::from_millis(500);

/// A validated client command for application-owned dispatch.
pub struct Request {
    pub id: String,
    pub session: String,
    pub client: String,
    pub command: String,
    pub args: Value,
    /// The application must check this before beginning command execution.
    pub deadline: Instant,
    pub reply: Sender<Value>,
}

/// Running loopback transport. Dropping it stops accepting clients and joins
/// the listener and its short-lived, bounded client handlers.
pub struct Server {
    pub rx: Receiver<Request>,
    running: Arc<AtomicBool>,
    listener: Option<JoinHandle<()>>,
    handlers: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    id: String,
    session: String,
    client: String,
    command: String,
    args: Value,
}

/// Starts the fixed, loopback-only transport.
pub fn start() -> Result<Server, String> {
    start_at(ADDRESS).map(|(server, _)| server)
}

fn start_at(address: &str) -> Result<(Server, SocketAddr), String> {
    let requested_address: SocketAddr = address
        .parse()
        .map_err(|_| "MCP transport address must be a numeric loopback address".to_owned())?;
    if !requested_address.ip().is_loopback() {
        return Err("MCP transport must bind to a loopback address".to_owned());
    }
    let listener = TcpListener::bind(requested_address)
        .map_err(|error| format!("Could not bind MCP loopback transport: {error}"))?;
    let bound_address = listener
        .local_addr()
        .map_err(|error| format!("Could not read MCP listener address: {error}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("Could not configure MCP listener: {error}"))?;

    let (tx, rx) = mpsc::sync_channel(QUEUE_CAPACITY);
    let running = Arc::new(AtomicBool::new(true));
    let handlers = Arc::new(Mutex::new(Vec::new()));
    let active_clients = Arc::new(AtomicUsize::new(0));
    let listener_running = Arc::clone(&running);
    let listener_handlers = Arc::clone(&handlers);

    let listener_thread = thread::Builder::new()
        .name("kraken-mcp-listener".to_owned())
        .spawn(move || {
            while listener_running.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        reap_finished_handlers(&listener_handlers);
                        if !reserve_client(&active_clients) {
                            let _ =
                                write_reply(stream, error_reply("", "Too many local MCP clients."));
                            continue;
                        }

                        let handler_tx = tx.clone();
                        let handler_running = Arc::clone(&listener_running);
                        let handler_active = Arc::clone(&active_clients);
                        let handle = thread::Builder::new()
                            .name("kraken-mcp-client".to_owned())
                            .spawn(move || {
                                handle_client(stream, handler_tx, handler_running);
                                handler_active.fetch_sub(1, Ordering::AcqRel);
                            });
                        if let Ok(handle) = handle {
                            let mut handles = listener_handlers
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            handles.push(handle);
                        } else {
                            active_clients.fetch_sub(1, Ordering::AcqRel);
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        reap_finished_handlers(&listener_handlers);
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => thread::sleep(Duration::from_millis(20)),
                }
            }
        })
        .map_err(|error| format!("Could not start MCP listener: {error}"))?;

    Ok((
        Server {
            rx,
            running,
            listener: Some(listener_thread),
            handlers,
        },
        bound_address,
    ))
}

fn reap_finished_handlers(handles: &Mutex<Vec<JoinHandle<()>>>) {
    let mut handles = handles
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut pending = Vec::with_capacity(handles.len());
    for handle in handles.drain(..) {
        if handle.is_finished() {
            let _ = handle.join();
        } else {
            pending.push(handle);
        }
    }
    *handles = pending;
}

fn reserve_client(active: &AtomicUsize) -> bool {
    let mut current = active.load(Ordering::Acquire);
    loop {
        if current >= MAX_CLIENTS {
            return false;
        }
        match active.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(next) => current = next,
        }
    }
}

fn handle_client(mut stream: TcpStream, tx: SyncSender<Request>, running: Arc<AtomicBool>) {
    // Windows may inherit the listener's nonblocking mode on accepted sockets.
    // A just-connected client can send after accept; wait using the I/O timeout
    // rather than treating WouldBlock as a malformed request and resetting it.
    if stream.set_nonblocking(false).is_err() {
        return;
    }
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let message = match read_message(&mut stream) {
        Ok(message) => message,
        Err(error) => {
            let _ = write_reply(stream, error_reply("", &error));
            return;
        }
    };

    // Raw HTTP is never a command channel. Reject it before parsing or queueing.
    if message.starts_with(b"GET ")
        || message.starts_with(b"POST ")
        || message.starts_with(b"PUT ")
        || message.starts_with(b"DELETE ")
        || message.starts_with(b"HEAD ")
        || message.starts_with(b"OPTIONS ")
    {
        let _ = write_reply(
            stream,
            error_reply("", "HTTP is not accepted by this endpoint."),
        );
        return;
    }

    let envelope: Envelope = match serde_json::from_slice(&message) {
        Ok(envelope) => envelope,
        Err(_) => {
            let _ = write_reply(stream, error_reply("", "Invalid request envelope."));
            return;
        }
    };
    if let Err(message) = validate_envelope(&envelope) {
        let _ = write_reply(stream, error_reply(&envelope.id, message));
        return;
    }

    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let id = envelope.id.clone();
    let (reply_tx, reply_rx) = mpsc::channel();
    let request = Request {
        id: envelope.id,
        session: envelope.session,
        client: envelope.client,
        command: envelope.command,
        args: envelope.args,
        deadline,
        reply: reply_tx,
    };

    match tx.try_send(request) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            let _ = write_reply(stream, error_reply(&id, "MCP command queue is full."));
            return;
        }
        Err(TrySendError::Disconnected(_)) => {
            let _ = write_reply(stream, error_reply(&id, "MCP dispatcher is unavailable."));
            return;
        }
    }

    loop {
        if !running.load(Ordering::Acquire) {
            return;
        }
        let now = Instant::now();
        if now >= deadline {
            let _ = write_reply(
                stream,
                error_reply(
                    &id,
                    "Command acknowledgement timed out; outcome is unknown.",
                ),
            );
            return;
        }
        let wait = (deadline - now).min(Duration::from_millis(100));
        match reply_rx.recv_timeout(wait) {
            Ok(reply) => {
                let _ = write_reply(stream, reply);
                return;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = write_reply(stream, error_reply(&id, "MCP dispatcher stopped."));
                return;
            }
        }
    }
}

fn validate_envelope(envelope: &Envelope) -> Result<(), &'static str> {
    validate_field(&envelope.id, 256, "id")?;
    validate_field(&envelope.session, 256, "session")?;
    validate_field(&envelope.client, 256, "client")?;
    validate_field(&envelope.command, 128, "command")?;
    Ok(())
}

fn validate_field(value: &str, maximum: usize, name: &'static str) -> Result<(), &'static str> {
    if value.trim().is_empty() {
        return Err(match name {
            "id" => "Request id is required.",
            "session" => "Session is required.",
            "client" => "Client is required.",
            _ => "Command is required.",
        });
    }
    if value.len() > maximum {
        return Err(match name {
            "id" => "Request id is too long.",
            "session" => "Session is too long.",
            "client" => "Client is too long.",
            _ => "Command is too long.",
        });
    }
    Ok(())
}

fn read_message(stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    let mut reader = BufReader::new(stream);
    let mut message = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if Instant::now() >= deadline {
            return Err("Request read deadline exceeded.".to_owned());
        }
        let available = reader
            .fill_buf()
            .map_err(|_| "Request read failed or timed out.".to_owned())?;
        if available.is_empty() {
            return Err("Request is empty.".to_owned());
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(available.len(), |position| position + 1);
        if message.len() + take > MAX_MESSAGE_BYTES {
            return Err("Request exceeds the 1 MiB limit.".to_owned());
        }
        message.extend_from_slice(&available[..take]);
        reader.consume(take);
        if newline.is_some() {
            while matches!(message.last(), Some(b'\n' | b'\r')) {
                message.pop();
            }
            return if message.is_empty() {
                Err("Request is empty.".to_owned())
            } else {
                Ok(message)
            };
        }
    }
}

fn write_reply(mut stream: TcpStream, reply: Value) -> io::Result<()> {
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let reply_id = reply
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| id.len() <= 256)
        .unwrap_or("");
    let serialized = serde_json::to_vec(&reply).unwrap_or_else(|_| {
        b"{\"id\":\"\",\"ok\":false,\"error\":\"Could not serialize reply.\"}".to_vec()
    });
    let response = if serialized.len() >= MAX_MESSAGE_BYTES {
        serde_json::to_vec(&error_reply(reply_id, "Response exceeds the 1 MiB limit."))
            .expect("static reply serializes")
    } else {
        serialized
    };
    stream.write_all(&response)?;
    stream.write_all(b"\n")?;
    stream.flush()
}

fn error_reply(id: &str, message: &str) -> Value {
    json!({
        "id": id,
        "ok": false,
        "error": message,
    })
}

impl Drop for Server {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(listener) = self.listener.take() {
            let _ = listener.join();
        }
        let mut handles = self
            .handlers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for handle in handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn request_line(id: &str) -> String {
        format!(
            "{{\"id\":\"{id}\",\"session\":\"s\",\"client\":\"test\",\"command\":\"status\",\"args\":{{}}}}\n"
        )
    }

    #[test]
    fn delivers_valid_loopback_request_and_reply() {
        let (server, address) = start_at("127.0.0.1:0").expect("server starts");
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).expect("connect");
            stream
                .write_all(request_line("one").as_bytes())
                .expect("write");
            let mut reply = String::new();
            stream.read_to_string(&mut reply).expect("read reply");
            reply
        });
        let request = server
            .rx
            .recv_timeout(Duration::from_secs(2))
            .expect("request arrives");
        assert_eq!(request.id, "one");
        assert!(request.deadline > Instant::now());
        request
            .reply
            .send(json!({"id":"one","ok":true,"result":{}}))
            .expect("reply sent");
        assert!(client
            .join()
            .expect("client joined")
            .contains("\"ok\":true"));
    }

    #[test]
    fn accepts_a_client_that_sends_after_connecting() {
        let (server, address) = start_at("127.0.0.1:0").expect("server starts");
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).expect("connect");
            thread::sleep(Duration::from_millis(100));
            stream
                .write_all(request_line("delayed").as_bytes())
                .expect("write after connect");
            let mut reply = String::new();
            stream.read_to_string(&mut reply).expect("read reply");
            reply
        });
        let request = server
            .rx
            .recv_timeout(Duration::from_secs(2))
            .expect("delayed request arrives");
        assert_eq!(request.id, "delayed");
        request
            .reply
            .send(json!({"id":"delayed","ok":true,"result":{}}))
            .expect("reply sent");
        assert!(client
            .join()
            .expect("client joined")
            .contains("\"ok\":true"));
    }

    #[test]
    fn serves_many_sequential_connections() {
        let (server, address) = start_at("127.0.0.1:0").expect("server starts");
        for number in 0..30 {
            let id = format!("sequence-{number}");
            let mut stream = TcpStream::connect(address).expect("connect");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("set read timeout");
            stream
                .write_all(request_line(&id).as_bytes())
                .expect("write request");
            let request = server
                .rx
                .recv_timeout(Duration::from_secs(2))
                .expect("request arrives");
            assert_eq!(request.id, id);
            request
                .reply
                .send(json!({"id": id, "ok": true, "result": {}}))
                .expect("reply sent");
            let mut reply = String::new();
            stream.read_to_string(&mut reply).expect("read reply");
            assert!(reply.contains("\"ok\":true"));
        }
    }

    #[test]
    fn rejects_oversize_envelope_before_dispatch() {
        let (server, address) = start_at("127.0.0.1:0").expect("server starts");
        let mut stream = TcpStream::connect(address).expect("connect");
        stream
            .write_all(&vec![b'x'; MAX_MESSAGE_BYTES + 1])
            .expect("write");
        stream.write_all(b"\n").expect("newline");
        let mut reply = String::new();
        stream.read_to_string(&mut reply).expect("read reply");
        assert!(reply.contains("Request exceeds the 1 MiB limit."));
        assert!(matches!(
            server.rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
    }
}
