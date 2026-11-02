from __future__ import annotations

import asyncio
import json
import os
import queue
import socket
import subprocess
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import KrakenBridge, KrakenBridgeError, ROOT


class FakeApp:
    def __init__(self, *, root=ROOT, lose_reply_command=None):
        self.calls = []
        self.root, self.lose_reply_command = root, lose_reply_command
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.port = self.listener.getsockname()[1]
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
    def _serve(self):
        while not self.stop.is_set():
            self.listener.settimeout(.1)
            try: conn, _ = self.listener.accept()
            except (TimeoutError, OSError): continue
            with conn:
                data = conn.recv(65536)
                request = json.loads(data.split(b"\n", 1)[0])
                self.calls.append(request)
                if request["command"] == self.lose_reply_command:
                    continue
                result = ({"application": "kraken-upscale", "project_root": str(self.root), "pid": 1}
                          if request["command"] in {"connect", "get_state"} else {"accepted": request["command"]})
                conn.sendall((json.dumps({"id": request["id"], "ok": True, "result": result}) + "\n").encode())
    def close(self):
        self.stop.set(); self.listener.close(); self.thread.join(1)


class BridgeTests(unittest.TestCase):
    def setUp(self): self.app = FakeApp(); self.bridge = KrakenBridge(port=self.app.port)
    def tearDown(self): self.app.close()
    def async_run(self, awaitable): return asyncio.run(awaitable)
    def test_lifecycle_and_command_protocol(self):
        self.async_run(self.bridge.connect())
        result = self.async_run(self.bridge.command("configure", {"settings": {"scale": 2}}))
        self.async_run(self.bridge.disconnect())
        self.assertEqual(result["accepted"], "configure")
        self.assertEqual([item["command"] for item in self.app.calls], ["connect", "configure", "disconnect"])
        self.assertTrue(all(isinstance(item["id"], str) and isinstance(item["session"], str) for item in self.app.calls))
        self.assertEqual(self.app.calls[0]["client"], "MCP client")
    def test_client_name_reconnects_before_command(self):
        self.async_run(self.bridge.use_client("automation-test"))
        self.async_run(self.bridge.command("preview"))
        self.assertEqual([call["command"] for call in self.app.calls], ["connect", "preview"])
        self.assertTrue(all(call["client"] == "automation-test" for call in self.app.calls))
    def test_no_command_without_session(self):
        with self.assertRaisesRegex(KrakenBridgeError, "launch_app"):
            self.async_run(self.bridge.command("upscale"))
        self.assertEqual(self.app.calls, [])
    def test_failed_mutation_is_not_replayed(self):
        self.app.close()
        self.app = FakeApp(lose_reply_command="upscale")
        bridge = KrakenBridge(port=self.app.port)
        with self.assertRaisesRegex(KrakenBridgeError, "outcome is unknown"):
            self.async_run(bridge._exchange("upscale"))
        self.assertEqual([call["command"] for call in self.app.calls], ["upscale"])
    def test_connect_rejects_a_different_project(self):
        self.app.close()
        self.app = FakeApp(root=ROOT.parent)
        bridge = KrakenBridge(port=self.app.port)
        with self.assertRaisesRegex(KrakenBridgeError, "different project"):
            self.async_run(bridge.connect())
    @unittest.skipUnless(sys.platform == "win32", "Windows app launcher")
    def test_explorer_handoff_uses_fixed_app_and_isolates_stdio(self):
        with patch("server.subprocess.Popen") as launch:
            self.bridge._launch_via_explorer_sync()
        arguments, options = launch.call_args
        self.assertEqual(arguments[0][1:], [str(ROOT / "Kraken Upscale.exe")])
        self.assertTrue(arguments[0][0].lower().endswith("\\explorer.exe"))
        options = launch.call_args.kwargs
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertEqual(options["stdout"], subprocess.DEVNULL)
        self.assertEqual(options["stderr"], subprocess.DEVNULL)
        self.assertTrue(options["close_fds"])
    @unittest.skipUnless(sys.platform == "win32", "Windows Explorer fallback")
    def test_start_uses_explorer_handoff_without_direct_app_process(self):
        with patch.object(KrakenBridge, "_launch_via_explorer_sync") as broker, \
             patch("server.subprocess.Popen") as launch:
            self.async_run(self.bridge._start_visible_app())
        broker.assert_called_once_with()
        launch.assert_not_called()


class StdioProtocolTests(unittest.TestCase):
    def test_real_stdio_tool_discovery(self):
        """Exercise the official SDK's actual stdio JSON-RPC transport."""
        runtime = os.environ.get("MCP_TEST_PYTHON", sys.executable)
        if subprocess.run([runtime, "-c", "import mcp"], capture_output=True).returncode:
            self.skipTest("MCP SDK is not installed in the selected test Python")
        server = Path(__file__).resolve().parents[1] / "server.py"
        process = subprocess.Popen([runtime, str(server)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8")
        try:
            replies: queue.Queue[str | None] = queue.Queue()
            def read_stdout():
                assert process.stdout
                for line in process.stdout:
                    replies.put(line)
                replies.put(None)
            threading.Thread(target=read_stdout, daemon=True).start()
            def request(item):
                assert process.stdin
                process.stdin.write(json.dumps(item) + "\n"); process.stdin.flush()
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    try: line = replies.get(timeout=max(0.01, deadline - time.monotonic()))
                    except queue.Empty: break
                    if line is None: break
                    reply = json.loads(line)
                    if reply.get("id") == item.get("id"): return reply
                self.fail("stdio MCP server did not answer")
            initialize = request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test-client", "version": "1"}}})
            self.assertIn("result", initialize)
            assert process.stdin
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"); process.stdin.flush()
            listed = request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            names = {tool["name"] for tool in listed["result"]["tools"]}
            self.assertTrue({"launch_app", "get_state", "upscale", "get_logs"}.issubset(names))
        finally:
            process.terminate()
            try: process.wait(3)
            except subprocess.TimeoutExpired: process.kill()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream: stream.close()

if __name__ == "__main__": unittest.main()
