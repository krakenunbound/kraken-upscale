"""Opt-in live verification through the real stdio MCP tools, never a renderer.

Default: launch/reuse the app, inspect schemas and show its MCP panel.
--exercise-media PATH additionally imports and exports a COPY of that fixture
inside MCP/.smoke-output, and restores the original idle app settings afterward.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Implementation

ROOT = Path(__file__).resolve().parent.parent


async def run(media: Path | None, hold: int) -> None:
    parameters = StdioServerParameters(command=sys.executable, args=[str(ROOT / "MCP/server.py")], cwd=ROOT)
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer, client_info=Implementation(name="Kraken MCP verification", version="1")) as session:
            await session.initialize()

            async def call(name: str, arguments: dict | None = None, expect_error: bool = False) -> dict:
                result = await asyncio.wait_for(session.call_tool(name, arguments or {}), timeout=40)
                if expect_error:
                    assert result.isError, f"{name} should have rejected invalid input"
                    return {}
                if result.isError:
                    raise RuntimeError(f"{name}: {result.content}")
                value = result.structuredContent
                if value is None:
                    value = json.loads(next(item.text for item in result.content if item.type == "text"))
                print(name, "OK", flush=True)
                return value

            launched = await call("launch_app")
            original = await call("get_state")
            schema = await call("describe_controls")
            assert "rtx_vsr_quality_index" in schema["settings"]
            await call("open_panel", {"panel": "mcp"})
            if media:
                assert not original["has_media"] and not original["tasks"] and not original["export_running"], "Media test requires an idle, empty app; existing work was preserved"
                fixture_root = ROOT / "MCP/.smoke-output"
                fixture_root.mkdir(exist_ok=True)
                fixture = fixture_root / media.name
                shutil.copy2(media, fixture)
                mode = "image" if media.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} else "video"
                await call("set_mode", {"mode": mode})
                defaults = await call("get_state")
                configured = {"resolution_index": 4, "rtx_vsr_quality_index": 3}
                configured["image_pipeline_index" if mode == "image" else "video_pipeline_index"] = 2 if mode == "image" else 1
                if mode == "video": configured["interp_on"] = False
                await call("configure", {"settings": configured})
                before = await call("get_state")
                await call("configure", {"settings": {"resolution_index": 999, "strength": 0.25}}, expect_error=True)
                after = await call("get_state")
                assert before["settings"] == after["settings"], "Invalid settings changed the form"
                await call("import_media", {"path": str(fixture)})
                await call("set_view", {"settings": {"view_mode": 1, "zoom": 1.0, "split": 0.5}})
                if mode == "video":
                    await call("playback", {"playing": True})
                    await asyncio.sleep(1)
                    await call("playback", {"playing": False})
                    await call("seek", {"position": 0.5})
                await call("preview")
                await call("add_to_batch")
                started = await call("batch_upscale")
                assert started["tasks"], "App did not create a queue entry"
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    await asyncio.sleep(2)
                    state = await call("get_state")
                    if state["tasks"] and all(row["done"] or row["failed"] for row in state["tasks"]): break
                else: raise RuntimeError("Export still running; inspect the app. It was not cancelled.")
                assert all(row["done"] and not row["failed"] for row in state["tasks"]), state["tasks"]
                for task in state["tasks"]:
                    output = Path(task["path"])
                    assert output.is_file() and output.stat().st_size > 0, task
                    print("Verified output:", output, flush=True)
                (fixture_root / "result.local.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
                await call("clear_queue")
                await call("clear_media")
                # Restore only controls changed by this test, in the tested mode.
                await call("configure", {"settings": {name: defaults["settings"][name] for name in configured}})
                await call("set_mode", {"mode": original["mode"]})
                await call("set_view", {"settings": original["view"]})
                await call("release_gpu_memory")
            await call("open_panel", {"panel": "mcp"})
            logs = await call("get_logs", {"kind": "mcp", "limit": 100})
            assert logs["lines"], "No app audit records"
            print("LIVE MCP CHECK PASSED; holding connection for visual inspection", hold, "seconds", flush=True)
            await asyncio.sleep(hold)
    # Start a NEW MCP session and query only; launch_app would hide a restart.
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer, client_info=Implementation(name="Kraken reconnect check", version="1")) as session:
            await session.initialize()
            result = await asyncio.wait_for(session.call_tool("get_state", {}), timeout=30)
            assert not result.isError, "App did not survive MCP disconnection"
            state = result.structuredContent or json.loads(next(item.text for item in result.content if item.type == "text"))
            assert state["pid"] == original["pid"], "App restarted after MCP disconnection"
            if media:
                assert state["gpu"]["device"] != "—", "GPU telemetry unavailable after agent launch"
    print("Verified: MCP disconnected/reconnected and the same app remains open.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise-media", type=Path)
    parser.add_argument("--hold-seconds", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args.exercise_media.resolve() if args.exercise_media else None, max(0, args.hold_seconds)))
