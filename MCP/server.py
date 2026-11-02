"""Stdio MCP bridge for the visible Kraken Upscale desktop application.

The bridge deliberately exposes only the desktop application's local control
protocol.  It does not process media, edit files, or provide shell access.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "Kraken Upscale.exe"
LAUNCH_LOG = ROOT / "MCP" / "kraken-upscale-launch.log"
HOST = "127.0.0.1"
PORT = 9756
TIMEOUT_SECONDS = 20.0
MAX_RESPONSE_BYTES = 1_048_576
LEASE_SECONDS = 12
HEARTBEAT_SECONDS = 3


class KrakenBridgeError(RuntimeError):
    """An actionable error returned to an MCP caller."""


def _validate_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise KrakenBridgeError(f"{label} must be a JSON object")
    return value


class KrakenBridge:
    """Fixed local TCP client. One request uses one socket and is never retried."""

    def __init__(self, *, host: str = HOST, port: int = PORT, root: Path = ROOT) -> None:
        self.host, self.port, self.root = host, port, root
        self.session_id = str(uuid.uuid4())
        self.client_name = "MCP client"
        self.connected = False

    def _validate_identity(self, state: dict[str, Any]) -> None:
        if state.get("application") != "kraken-upscale":
            raise KrakenBridgeError("The local control bridge does not belong to Kraken Upscale.")
        project_root = state.get("project_root")
        if not isinstance(project_root, str):
            raise KrakenBridgeError("Kraken Upscale did not report its project root.")
        if Path(project_root).resolve(strict=False) != self.root.resolve(strict=False):
            raise KrakenBridgeError("The local Kraken Upscale control bridge belongs to a different project folder.")

    def _exchange_sync(self, command: str, args: dict[str, Any]) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        payload = {
            "id": request_id,
            "session": self.session_id,
            "client": self.client_name,
            "command": command,
            "args": args,
        }
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            stream = socket.create_connection((self.host, self.port), timeout=TIMEOUT_SECONDS)
        except OSError as exc:
            raise KrakenBridgeError("Kraken Upscale is not connected. Use launch_app to open the visible desktop app.") from exc
        sent = False
        try:
            with stream:
                stream.settimeout(TIMEOUT_SECONDS)
                stream.sendall(encoded)
                sent = True
                response = bytearray()
                while len(response) <= MAX_RESPONSE_BYTES:
                    chunk = stream.recv(min(65536, MAX_RESPONSE_BYTES + 1 - len(response)))
                    if not chunk:
                        break
                    response.extend(chunk)
                    if b"\n" in chunk:
                        break
        except OSError as exc:
            if sent:
                raise KrakenBridgeError("Kraken Upscale did not confirm this command; its outcome is unknown. Inspect get_state and get_logs before issuing another command. The command was not retried.") from exc
            raise KrakenBridgeError("Kraken Upscale could not accept this command. Use launch_app to open the visible desktop app.") from exc
        if len(response) > MAX_RESPONSE_BYTES:
            raise KrakenBridgeError("Kraken Upscale returned a response larger than 1 MB")
        line = bytes(response).split(b"\n", 1)[0]
        try:
            reply = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KrakenBridgeError("Kraken Upscale did not confirm this command; its outcome is unknown. Inspect get_state and get_logs before issuing another command.") from exc
        if not isinstance(reply, dict) or reply.get("id") != request_id:
            raise KrakenBridgeError("Kraken Upscale did not confirm this command; its outcome is unknown. Inspect get_state and get_logs before issuing another command.")
        if reply.get("ok") is not True:
            detail = reply.get("error")
            raise KrakenBridgeError(str(detail) if isinstance(detail, str) else "Kraken Upscale rejected the command")
        result = reply.get("result")
        if not isinstance(result, dict):
            raise KrakenBridgeError("Kraken Upscale returned an invalid command result")
        return result

    async def _exchange(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._exchange_sync, command, args or {})

    async def connect(self) -> None:
        state = await self._exchange("connect", {"lease_seconds": LEASE_SECONDS})
        self._validate_identity(state)
        self.connected = True

    def adopt_client(self, name: str) -> None:
        if name and name != self.client_name:
            self.client_name = name[:200]
            self.connected = False

    async def use_client(self, name: str) -> None:
        self.adopt_client(name)
        if not self.connected:
            await self.connect()

    async def disconnect(self) -> None:
        if not self.connected:
            return
        try:
            await self._exchange("disconnect", {})
        finally:
            self.connected = False

    async def heartbeat(self) -> None:
        if self.connected:
            await self._exchange("heartbeat", {"lease_seconds": LEASE_SECONDS})

    async def require_app(self) -> None:
        if not self.connected:
            raise KrakenBridgeError("Kraken Upscale is not connected. Use launch_app to open the visible desktop app.")

    async def command(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        await self.require_app()
        return await self._exchange(command, args)

    @staticmethod
    def _existing_app_process() -> bool:
        if sys.platform != "win32":
            return False
        completed = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Kraken Upscale.exe", "/NH"], capture_output=True, text=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
        return "Kraken Upscale.exe" in completed.stdout

    @staticmethod
    def _open_launch_log():
        """Keep child diagnostics local and bounded, never on MCP stdout/stderr."""
        LAUNCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            if LAUNCH_LOG.exists() and LAUNCH_LOG.stat().st_size > 512 * 1024:
                with LAUNCH_LOG.open("rb") as source:
                    source.seek(-256 * 1024, os.SEEK_END)
                    tail = source.read()
                LAUNCH_LOG.write_bytes(b"[older launch output truncated]\r\n" + tail)
            return LAUNCH_LOG.open("ab", buffering=0)
        except OSError as exc:
            raise KrakenBridgeError("Could not open the local Kraken Upscale launch diagnostic log") from exc

    def _launch_via_explorer_sync(self) -> None:
        """Hand the fixed app path to the existing interactive Explorer process.

        Shell.Application created in this process is NOT an out-of-process
        launch broker: it can leave the app in the MCP job. Explorer's command
        handoff instead lets the user's desktop own the app and its environment.
        """
        try:
            explorer = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "explorer.exe"
            subprocess.Popen([str(explorer), str(APP_PATH)], cwd=str(self.root),
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, close_fds=True)
        except Exception as exc:
            raise KrakenBridgeError("Could not ask the Windows desktop to launch Kraken Upscale.") from exc

    async def _start_visible_app(self) -> None:
        if sys.platform != "win32":
            raise KrakenBridgeError("Kraken Upscale can only be launched independently through this Windows MCP package.")
        log = self._open_launch_log()
        try:
            # Ask the existing desktop shell to own the app independently.
            log.write(b"[launching through Explorer desktop broker]\r\n")
            await asyncio.to_thread(self._launch_via_explorer_sync)
        finally:
            log.close()

    async def launch_app(self) -> dict[str, Any]:
        """Reuse an app that has the control bridge, or visibly start the packaged app."""
        try:
            await self.connect()
            state = await self._exchange("get_state", {})
            self._validate_identity(state)
            return {"launched": False, "reused": True, "state": state}
        except KrakenBridgeError:
            self.connected = False
        if self._existing_app_process():
            raise KrakenBridgeError("A Kraken Upscale process is already running but does not expose this matching control bridge. Close it or restart the current app; no second visible app was launched.")
        if not APP_PATH.is_file():
            raise KrakenBridgeError(f"Kraken Upscale executable was not found at {APP_PATH}")
        await self._start_visible_app()
        deadline = time.monotonic() + TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                await self.connect()
                state = await self._exchange("get_state", {})
                self._validate_identity(state)
                if state.get("application") == "kraken-upscale":
                    return {"launched": True, "reused": False, "state": state}
                await self.disconnect()
                last_error = KrakenBridgeError("The app control bridge did not identify Kraken Upscale")
            except KrakenBridgeError as exc:
                last_error = exc
            await asyncio.sleep(0.25)
        raise KrakenBridgeError("Kraken Upscale was started but did not open its control bridge within 20 seconds") from last_error


def _annotations(read_only: bool = False, destructive: bool = False):
    from mcp.types import ToolAnnotations
    return ToolAnnotations(readOnlyHint=read_only, destructiveHint=destructive, idempotentHint=read_only, openWorldHint=False)


def create_mcp_server(bridge: KrakenBridge | None = None):
    """Create a testable official MCP SDK server over stdio."""
    try:
        from mcp.server.fastmcp import Context, FastMCP
    except ModuleNotFoundError as exc:
        raise RuntimeError("MCP SDK is not installed. Run MCP/setup.ps1 first.") from exc
    client = bridge or KrakenBridge()
    # FastMCP evaluates postponed annotations against this module's globals.
    globals()["Context"] = Context

    @asynccontextmanager
    async def lifespan(_server):
        # An app need not be running when an MCP client initializes. The first
        # ordinary tool call fails safely until launch_app establishes its lease.
        try:
            await client.connect()
        except KrakenBridgeError:
            pass
        async def beat() -> None:
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                try:
                    await client.heartbeat()
                except KrakenBridgeError:
                    client.connected = False
        task = asyncio.create_task(beat())
        try:
            yield {"bridge": client}
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            with suppress(KrakenBridgeError):
                await client.disconnect()

    mcp = FastMCP("Kraken Upscale", instructions="Controls the visible Kraken Upscale desktop app only.", lifespan=lifespan)
    read, action, destructive = _annotations(True), _annotations(), _annotations(destructive=True)

    def tool(name: str, description: str, annotation=action):
        return mcp.tool(name=name, description=description, annotations=annotation)

    async def prepare(ctx: Context, *, connect: bool = True) -> None:
        params = ctx.session.client_params
        info = getattr(params, "clientInfo", None)
        name = getattr(info, "name", None)
        name = name if isinstance(name, str) and name else "MCP client"
        if connect:
            await client.use_client(name)
        else:
            client.adopt_client(name)

    @tool("launch_app", "Open the visible Kraken Upscale application or reuse its matching app control bridge.")
    async def launch_app(ctx: Context) -> dict[str, Any]:
        await prepare(ctx, connect=False); return await client.launch_app()

    @tool("get_state", "Read the state of the visible Kraken Upscale application.", read)
    async def get_state(ctx: Context) -> dict[str, Any]:
        await prepare(ctx); return await client.command("get_state")

    @tool("describe_controls", "Read the app-provided controls and accepted settings schema.", read)
    async def describe_controls(ctx: Context) -> dict[str, Any]:
        await prepare(ctx); return await client.command("describe_controls")

    @tool("configure", "Apply settings in the visible app before processing.")
    async def configure(settings: dict[str, Any], ctx: Context) -> dict[str, Any]:
        await prepare(ctx); return await client.command("configure", {"settings": _validate_object(settings, "settings")})

    @tool("import_media", "Import a media file into the visible app.")
    async def import_media(path: str, ctx: Context) -> dict[str, Any]:
        await prepare(ctx); return await client.command("import_media", {"path": path})

    @tool("set_mode", "Select video or image mode in the visible app.")
    async def set_mode(mode: str, ctx: Context) -> dict[str, Any]:
        await prepare(ctx)
        if mode not in {"video", "image"}: raise KrakenBridgeError("mode must be video or image")
        return await client.command("set_mode", {"mode": mode})

    @tool("open_panel", "Open a named visible app panel, or close panels.")
    async def open_panel(panel: str, ctx: Context) -> dict[str, Any]:
        await prepare(ctx)
        if panel not in {"logs", "system", "input", "enhance", "motion", "output", "mcp", "models", "closed"}: raise KrakenBridgeError("invalid panel")
        return await client.command("open_panel", {"panel": panel})

    @tool("preview", "Refresh the visible preview.")
    async def preview(ctx: Context) -> dict[str, Any]:
        await prepare(ctx); return await client.command("preview")

    @tool("playback", "Start or pause visible preview playback.")
    async def playback(playing: bool, ctx: Context) -> dict[str, Any]:
        await prepare(ctx); return await client.command("playback", {"playing": playing})

    @tool("seek", "Seek the visible preview from 0.0 through 1.0.")
    async def seek(position: float, ctx: Context) -> dict[str, Any]:
        await prepare(ctx)
        if isinstance(position, bool) or not isinstance(position, (int, float)) or not 0 <= position <= 1: raise KrakenBridgeError("position must be a number from 0 through 1")
        return await client.command("seek", {"position": float(position)})

    @tool("set_view", "Change the visible preview comparison view.")
    async def set_view(settings: dict[str, Any], ctx: Context) -> dict[str, Any]:
        await prepare(ctx); return await client.command("set_view", {"settings": _validate_object(settings, "settings")})

    def _simple_tool(command: str):
        async def invoke(ctx: Context) -> dict[str, Any]:
            await prepare(ctx)
            return await client.command(command)
        return invoke

    for name, command, annotation, description in [
        ("add_to_batch", "add_to_batch", action, "Add the loaded media to the visible batch queue."),
        ("upscale", "upscale", action, "Start upscaling the loaded media in the visible app."),
        ("batch_upscale", "batch_upscale", action, "Start the visible app's batch upscale."),
        ("cancel_upscale", "cancel_upscale", destructive, "Cancel the current visible upscale only when requested."),
        ("clear_queue", "clear_queue", destructive, "Clear the visible batch queue."),
        ("clear_media", "clear_media", destructive, "Clear loaded media from the visible app."),
        ("release_gpu_memory", "release_gpu_memory", action, "Release Kraken Upscale model and interpolation caches."),
    ]:
        simple = _simple_tool(command)
        simple.__name__ = name
        tool(name, description, annotation)(simple)

    @tool("reveal_output", "Reveal one output item in the visible app.")
    async def reveal_output(index: int, ctx: Context) -> dict[str, Any]:
        await prepare(ctx)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0: raise KrakenBridgeError("index must be a non-negative whole number")
        return await client.command("reveal_output", {"index": index})

    @tool("get_logs", "Read bounded app or MCP log entries.", read)
    async def get_logs(ctx: Context, kind: str = "app", limit: int = 100) -> dict[str, Any]:
        await prepare(ctx)
        if kind not in {"app", "mcp"}: raise KrakenBridgeError("kind must be app or mcp")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500: raise KrakenBridgeError("limit must be a whole number from 1 through 500")
        return await client.command("get_logs", {"kind": kind, "limit": limit})

    @mcp.resource("kraken-upscale://skills")
    def skill_documents() -> str:
        docs = [
            ROOT / "MCP" / "skills" / "kraken-upscale" / "SKILL.md",
            ROOT / "MCP" / "AGENT-GUIDE.md",
        ]
        available = [path for path in docs if path.is_file()]
        if not available: return "Kraken Upscale skill documents have not been installed yet."
        return "\n\n".join(path.read_text(encoding="utf-8") for path in available)
    return mcp


def main() -> None:
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
