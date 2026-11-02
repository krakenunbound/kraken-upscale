---
name: kraken-upscale
description: Operate the running Kraken Upscale Windows app through its local MCP tools to import media, configure its visible upscaling controls, preview, queue, export, and inspect results. Use when a user asks to upscale video or images with their local Kraken Upscale installation.
---

# Kraken Upscale MCP

Use the `kraken-upscale` MCP server to control the running visible Kraken
Upscale desktop app. The app is the only processing authority: use MCP tools
that activate its controls and normal export actions. Do not substitute CLI
enhancement modes, FFmpeg, ONNX, NVIDIA RTX utilities, direct renderer scripts,
or settings-file edits for an MCP action.

Start every task with `launch_app`, `get_state`, and `describe_controls`.
`describe_controls` is authoritative for supported configuration field names,
values, installed models, and controls unavailable on this PC. Never invent
configuration names or indexes. To select a model, find its advertised name in
the ordered `models` array and pass the matching zero-based `model_index` to
`configure`; use `generative_models` and `generative_model_index` for a
generative package. `configure` does not accept model-name strings. Read
`get_state.runtime` before selecting optional RTX VSR, generative, or
interpolation features.

Set `video` or `image` mode before `import_media`, then inspect state after the
import. Use `configure` followed by state inspection. Request `preview` when a
preview would help assess the intended settings. `set_view`, `playback`, and
`seek` operate visible comparison/source-inspection controls; `seek` uses a
normalized `0..1` position.

`upscale` processes the first unfinished app queue item. If the queue is empty,
it creates a task from the loaded source. For a deliberate queue, use
`add_to_batch`, verify the queue in `get_state`, then call `batch_upscale`.
Poll `get_state` or `get_logs` instead of re-submitting. Queue rows have a
current `index`, not a stable job UUID; refresh state before using an index for
`reveal_output`. On completion, report the actual output paths returned by the
app. Outputs are automatically created beside the source with a unique suffix
if necessary; the app has no arbitrary output destination field.

Treat `cancel_upscale`, `clear_queue`, `clear_media`, and
`release_gpu_memory` as visible application actions. Do not cancel or clear
work without the user's instruction. Never release GPU memory while a render is
active. Use `reveal_output` only when the user wants Explorer opened.

The left MCP tab turns green while a bridge connection is active and logs
connection/command summaries. A bridge lease lasts 12 seconds and renews every
three seconds. Disconnection never stops a render. After a timeout or
reconnection, call `get_state` and inspect `get_logs` before retrying any
state-changing command; do not replay it automatically.

`Preserve grain` is currently marked planned in the UI, so do not describe it
as an active processing feature.

For the detailed, portable workflow, read
[`references/operation.md`](references/operation.md). When this skill is copied
to a personal skill directory, use the connected MCP resource
`kraken-upscale://skills` for the project-specific setup guide and current
server documents; do not rely on paths outside this skill folder.
