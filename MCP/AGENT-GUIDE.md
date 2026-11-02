# Operating Kraken Upscale through MCP

Kraken Upscale is a native Windows video and image upscaler. Its normal work
uses NVIDIA CUDA/ONNX runtime, with optional NVIDIA RTX Video Super Resolution,
frame interpolation, and optional generative refinement where installed. MCP
drives that running desktop application so a user can see the active controls,
queue, progress, and output alongside the agent.

## Start with state and live controls

Call `launch_app`, `get_state`, and `describe_controls` in that order. State
answers whether the runtime is ready, which models are installed, the selected
mode and media, visible task state, and actual output paths. Optional runtime
availability is in `get_state.runtime`. The controls description is the source
of truth for settings and choices on this particular machine. A GPU, driver,
installed model, or optional runtime can make a choice unavailable.

Use `open_panel` when the user needs to inspect a part of the app. It accepts
`logs`, `system`, `input`, `enhance`, `motion`, `output`, `mcp`, and `closed`.
The MCP panel provides a visible audit trail; the other panels expose the same
normal app controls the MCP bridge changes.

Do not operate Kraken by launching command-line enhancement modes, invoking
FFmpeg or ONNX/RTX utilities, writing settings files, or manipulating a queue
behind the app. The MCP control module invokes the application's existing UI
callbacks. This preserves what the user sees and lets the application retain
ownership of GPU memory, previews, exports, output naming, and cancellation.

## Media and configuration

Set the required mode with `set_mode(video)` or `set_mode(image)` before
`import_media(path)`. Import a local supported media path only after checking
the mode. Inspect `get_state` after import; it reports the source, whether it
loaded, and app-detected metadata.

Call `configure` only with fields and values provided by `describe_controls`.
The app may expose names that resemble its Slint underscore-style controls, but
agents must never infer a field name or index. To choose an installed model,
find its advertised name in the returned ordered `models` array and set its
zero-based `model_index`; use `generative_models` and
`generative_model_index` for a generative package. `configure` does not accept
a model-name string. Check state after configuration because a selected
pipeline can change which options apply.

Main feature areas exposed by the desktop app include:

- **Input:** source handling and content/profile choices.
- **Enhance:** model or NVIDIA RTX VSR pipeline selection, scaling/resolution,
  strength/detail controls, and optional generative refinement when available.
- **Motion:** stability/temporal controls, cut detection, and RIFE
  interpolation when video mode and its model/runtime permit them.
- **Output:** video container, NVENC codec/quality and audio handling for
  video; PNG, JPEG, or WebP choices for images.

`Preserve grain` is visibly marked planned in the app. Do not present it as an
effective processing option even if it appears in saved/displayed state.

Use `set_view` for comparison layout and supported view controls, and `preview`
to run the app's preview. Playback and seeking are only source-inspection tools:
`playback` takes a boolean and `seek` accepts a normalized `0..1` position.

## Export workflow

For one item, use this sequence:

1. Inspect controls and configure the loaded media.
2. Optionally call `preview` and inspect the preview/app logs.
3. Call `upscale` once. It processes the first unfinished queued task. Only
   when the queue is empty does the app create a task from the loaded source.
4. Poll `get_state` or `get_logs` at a reasonable interval. Do not submit a
   duplicate export while the original task may still be running.
5. On completion, report the task's actual output path from `get_state`.
   Task rows use their current queue `index`, not a stable job UUID; refresh
   state before `reveal_output(index)`. Open Explorer only if the user wants it.

For several intended sources, import each source and use `add_to_batch`, inspect
`get_state` to confirm the queue, then call `batch_upscale` once. Use
`clear_queue` only when the user wants queued work removed. `cancel_upscale`
affects the application export and should be used only at the user's request or
when required by their stated workflow.

Kraken writes outputs beside each source and allocates a nonconflicting suffix
when a name is taken. Report the path returned by the app; do not calculate or
promise a filename in advance.

## GPU memory and recovery

The native application tracks model and interpolation caches. Use
`release_gpu_memory` when the user requests it, before intentionally handing
the GPU to other work, or after a task that is fully finished. Do not call it
during a render. The System panel and `get_state` show runtime/GPU information
when it is available.

If a command fails or the MCP connection is interrupted, first call
`get_state` and `get_logs`. The application may have accepted the command and
still be processing it. An MCP disconnect does not cancel rendering. Do not
retry a mutating command automatically until state shows it was not accepted.

## Audit and connection behavior

The application marks an MCP connection active using a 12-second lease that is
renewed every three seconds. The left-side MCP tab turns green while at least
one bridge is active and records connection and command summaries. Use
`get_logs(kind="mcp")` to inspect that audit history, or
`get_logs(kind="app")` for normal application activity. Opening the MCP panel
is useful when the user wants to watch the same commands the agent is issuing.

See [README.md](README.md) for installation and the complete tool list.
