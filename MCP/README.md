# Kraken Upscale MCP

Kraken Upscale MCP lets a compatible MCP client operate the **running, visible
Kraken Upscale desktop app**. It launches the normal Slint application when
needed, changes the same controls a person uses, and starts the app's normal
preview and export actions. The app's **MCP** tab shows the connected agent and
recent command activity; its tab label turns green while a client connection is
alive.

This is a local stdio server. It does not open a network listener, upload media,
or create a second rendering pipeline. It is intended for trusted agents on the
same Windows PC.

## Install

From the project root, run:

```powershell
pwsh -ExecutionPolicy Bypass -File .\MCP\setup.ps1
```

Setup requires PowerShell 7+ and Python 3.11+. It creates `MCP\.venv`, installs
the MCP Python dependency in that environment, and writes ready-to-paste
configuration for this checkout: `MCP\client-config.local.json`,
`MCP\codex-mcp.toml.example`, `MCP\vscode-mcp.json.example`,
`MCP\claude-desktop-config.example.json`, and `MCP\cursor-mcp.json.example`.
Run it again after moving the project. It does not install GPU dependencies or
change Kraken settings.

By default setup does not register a client. It can register selected clients:
`-RegisterCodex`, `-RegisterClaude`, `-RegisterClaudeDesktop`,
`-RegisterCursor`, or `-RegisterVSCode -VSCodeConfigPath .vscode/mcp.json`.
`-RegisterAll` detects Codex, Claude Code, and existing default Claude Desktop
or Cursor configurations; VS Code still needs an explicit workspace config
path. JSON registration preserves unrelated entries and backs up an existing
file as `<config>.kraken-upscale-backup` before changing it.

Use the command printed by setup for your client. A representative stdio entry
is:

```json
{
  "mcpServers": {
    "kraken-upscale": {
      "command": "F:\\Kraken Media\\Video Upscale\\MCP\\.venv\\Scripts\\python.exe",
      "args": ["F:\\Kraken Media\\Video Upscale\\MCP\\server.py"]
    }
  }
}
```

Codex, Claude, Cursor, and VS Code each have their own MCP configuration
location and reload behavior; paste the generated server entry into the
appropriate client configuration and reconnect the client. Other clients can
use the same command when they support local stdio MCP servers. A client that
does not support MCP cannot use this integration.

## Agent skill

The reusable agent instructions are at
[`skills/kraken-upscale/SKILL.md`](skills/kraken-upscale/SKILL.md). Copy or
junction the `kraken-upscale` folder into a client skill directory when that
client supports skills. For example, Codex commonly discovers skills under
`%USERPROFILE%\.codex\skills`; preserve the directory structure so the linked
reference remains available.

Alternatively, use `-InstallSkills -SkillsPath <personal-skills-directory>` with
setup. This copies the portable skill and refuses to overwrite an existing one.

Build, protocol, installer, and live app checks are recorded in [TESTING.md](TESTING.md).

## How an agent works with Kraken

1. Call `launch_app`, then `get_state`. Launching opens the ordinary desktop
   app; it does not make a hidden worker.
2. Call `describe_controls` before `configure`. It returns the live accepted
   setting names, allowed values, installed model names, and disabled or
   unavailable controls. Do not guess configuration fields from this document.
   To select a model, locate its advertised name in the returned `models` or
   `generative_models` array and submit its zero-based `model_index` or
   `generative_model_index` through `configure`; model names are not accepted
   configuration values.
3. Set `video` or `image` mode, import a local file, and inspect the returned
   state. `import_media` uses Kraken's normal import path and returns source
   metadata once it is loaded.
4. Configure the visible controls, request `preview` if desired, and inspect
   state/logs before starting expensive work.
5. `upscale` processes the first unfinished visible queue item. If the queue is
   empty, the app creates a task from the loaded source. Use `add_to_batch`,
   verify the queue, then `batch_upscale` to process every unfinished queue
   item. Poll `get_state` or `get_logs` at a sensible interval.

Output paths are chosen by the app beside the source. Kraken makes a unique
name when that path already exists. There is no arbitrary output-path override
because the desktop app has no such control.

The running application owns all work. Never replace these tools with
`--maelstrom-enhance`, direct FFmpeg, ONNX, RTX, or model scripts, and do not
edit the app settings files to automate an export. Those bypass the visible
application and can leave the UI and agent state inconsistent.

## Tools

| Tool | Purpose |
| --- | --- |
| `launch_app` | Start or focus the visible Kraken app and wait until it can accept controls. |
| `get_state` | Read mode, media, runtime/models, visible settings, queue, task and output state. |
| `describe_controls` | Read the current configuration schema and available UI choices. |
| `configure` | Apply supported settings through the app's control callbacks. |
| `import_media` | Import one supported local video or still image through the app. |
| `set_mode` | Select `video` or `image`. |
| `open_panel` | Show `logs`, `system`, `input`, `enhance`, `motion`, `output`, `mcp`, or `closed`. |
| `preview` | Run Kraken's preview action for the loaded media. |
| `playback` / `seek` | Start or stop source playback, or seek from `0` to `1`. |
| `set_view` | Set supported comparison/view controls from its live schema. |
| `add_to_batch` / `clear_queue` | Add the loaded media to, or clear, the visible batch queue. |
| `upscale` / `batch_upscale` | Process the first unfinished queue item (or make one from loaded media if empty), or process all unfinished queued items. |
| `cancel_upscale` | Request cancellation of the active app export. |
| `clear_media` | Clear the loaded item through the app. |
| `release_gpu_memory` | Release Kraken's model and interpolation caches. |
| `reveal_output` | Reveal a completed output selected by its returned index. |
| `get_logs` | Read recent `app` or `mcp` log records, with a limit. |

Tool input schemas from the connected server are authoritative. In particular,
`configure` accepts only names supplied by `describe_controls`. Match an
advertised model name to its array position, then pass that zero-based position
as `model_index` or `generative_model_index`; it does not accept a model name.
`get_state.runtime` reports availability for optional RTX VSR, generative, and
interpolation features. Queue/task rows use their current `index`, not a stable
job UUID, so refresh state before acting on an index.

## Connection and lifecycle

Each MCP bridge has a 12-second connection lease and renews it every three
seconds. If a client closes or crashes, the **MCP** label returns to its normal
color after the lease expires. A disconnected client never stops an in-progress
render. Reconnect, call `get_state`, and continue from the app's reported task
state; do not automatically replay a state-changing command after a timeout or
disconnect.

The MCP tab is an inspection surface. It records connection changes and command
summaries, while `get_logs(kind="mcp")` supplies those records to an agent.
Use `get_logs(kind="app")` for app activity. Sensitive local settings are not
returned as credentials.

For a fuller operational workflow and feature notes, see
[`AGENT-GUIDE.md`](AGENT-GUIDE.md).

MCP transport background: [MCP stdio transport specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
and the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
