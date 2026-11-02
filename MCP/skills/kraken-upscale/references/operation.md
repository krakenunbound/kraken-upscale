# Kraken Upscale MCP operation reference

Use this reference after loading the `kraken-upscale` skill when a task needs
the detailed operating rules. It stays with the skill when the skill folder is
copied to a personal skill directory.

Kraken Upscale is a native Windows video and image upscaler. MCP controls its
running, visible Slint app through the app's existing callbacks. It must not be
replaced with `--maelstrom-enhance`, FFmpeg, ONNX, NVIDIA RTX utilities, direct
renderer scripts, or settings-file edits. That would bypass the visible app and
its control/queue state.

Start with `launch_app`, `get_state`, then `describe_controls`. `get_state`
reports source, settings, view, queue rows, runtime availability and output
paths. `describe_controls` supplies live setting/view schemas, model arrays,
panel names, and notes. Its model arrays are ordered: choose a model by finding
its currently advertised name, then use the matching zero-based `model_index`
or `generative_model_index` in `configure`. A name string is not valid for
either configuration field. Some saved controls are inactive under RTX VSR;
the live notes say which ones. Preserve grain is planned and disabled.

Set `video` or `image` before importing an absolute local media file. Import
may change the mode to match the file, so check state afterward. Playback and
seek are video-only source-inspection controls. `open_panel` can display logs,
system, input, enhance, motion, output, MCP, or no panel; Motion is unavailable
in image mode.

`upscale` processes the first unfinished queue row. If no queue row exists, it
creates one from the loaded source. `batch_upscale` processes all unfinished
rows. After `add_to_batch`, inspect the returned queue before calling either
export action. Poll state or logs instead of submitting a duplicate command.
Task rows use a mutable current `index`, not a job UUID. Refresh state before
calling `reveal_output(index)`. Kraken writes unique output names beside the
input; report the actual returned path, never predict one.

MCP connections have a 12-second lease with a three-second heartbeat. The
left-side MCP label turns green while a bridge is live and shows connection and
command records. A disconnect never stops rendering. After errors, timeouts or
reconnection, inspect `get_state` and `get_logs` before any state-changing
retry. Do not automatically replay a mutation. Use `release_gpu_memory` only
when no export is scheduled or running.
