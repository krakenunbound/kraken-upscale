# Kraken Upscale — bug list

Static code audit against the current tree, informed by [DECISIONS.md](DECISIONS.md)
and [NEXT-SESSION.md](NEXT-SESSION.md).

**Scope:** defects and misleading behaviour only. Remaining planned product work
such as IVTC and grain preservation stays in DECISIONS / NEXT-SESSION unless UI
copy claims it is already active.

**Method:** full pass over `src/*.rs` and the export/preview control surface in
`ui/app.slint`. No fixes applied in this pass — findings only.

Last updated: 2026-08-07

---

> **Status 2026-08-07:** every finding below is closed. B33 and B35–B42 were
> fixed; B34 was investigated and closed as not a defect. The descriptions are
> kept as the record of what was wrong and why.

### B43 — Detail stage was still an image upscaler applied frame by frame

- **Where:** model discovery, preview, and export detail inference
- **What:** RIFE supplied genuine frame interpolation and the temporal lock
  reduced shimmer afterward, but the neural detail model still consumed one
  frame at a time. Calling that equivalent to video super-resolution overstated
  what the pipeline did.
- **Impact:** neighboring frames could not contribute detail evidence inside
  the model; the lock could only retain or release already-invented texture.
- **Fix:** added a distinct five-dimensional, multi-frame model contract and a
  genuine bidirectional NanoVSR stage. Preview and export now feed five-frame
  windows, tile spatially, pad EOF safely, reset at hard cuts, and bypass the
  frame-model lock. Frame models remain supported as a compatibility mode.
- **Verification:** real CUDA tensor test passed; a seven-frame export made
  from `SpaceVoyage_2.mp4` produced all seven frames at the requested 2x output,
  exercising two windows and the padded EOF path.
- [x] B43 — genuine multi-frame video super-resolution path

## Findings (2026-08-07 audit) — all closed

### Critical / high

### B33 — GPU-missing status is overwritten by `nvidia-smi` probe
- **Where:** `src/main.rs` (startup after `cuda::prepare`)
- **What:** When `cuda::prepare()` fails, the UI is set to `GPU RUNTIME MISSING`,
  then immediately overwritten by `gpu::probe().runtime`, which reports
  `RUNTIME OK` whenever `nvidia-smi` sees a device — even if the bundled CUDA /
  cuDNN DLLs required by ONNX Runtime are absent.
- **Impact:** Title/runtime pill can claim a healthy GPU path while preview and
  export will refuse or fail once a session is built. Users get a false green
  state at launch.
- **Suggested fix:** Derive runtime state from `cuda_dir` (and later actual
  provider registration), not from `nvidia-smi` alone. Keep device name from
  smi; keep readiness from the ORT/CUDA payload check.

### B34 — Temporal lock uses RIFE intermediates as motion source
- **Where:** `src/export.rs` (`process_frame` + interpolation loop)
- **What:** With Stability on, motion is estimated from the `source` argument
  passed into `process_frame`. On the interpolation path that argument is often
  a RIFE-synthesized frame (or alternates between real and synthetic frames),
  not the original consecutive source pair.
- **Impact:** Enabling Stability together with Frame Interpolation estimates
  optical flow on invented intermediates. The lock can smear, ghost, or hold
  the wrong detail precisely when both “video” features are active.
- **Suggested fix:** Track the last *original* decoded pair for flow, run the
  lock against that motion field, and only advance the lock’s source history on
  real source frames (or warp lock state across interpolated output times).

---

### Medium

### B35 — Content profile does not suppress Restore despite UI copy
- **Where:** `src/media.rs` (`SourceTreatment::filters`); `ui/app.slint` (Source help)
- **What:** UI help says AI-generated video “skips deinterlace, telecine and
  deblock entirely.” Deinterlace is gated to content indices 2|3, but Restore
  (`deblock` / `bilateral`) always runs for any profile. Telecine is never
  applied for any profile. Content values 0, 1, and 4 are otherwise no-ops.
- **Impact:** Choosing “AI generated video” (or Film / Modern digital) still
  applies heavy deblock when Restore is non-None — opposite of the stated
  contract. Easy to damage clean AI/film sources unintentionally.
- **Suggested fix:** Gate Restore (and any future IVTC) on broadcast-era
  profiles, or change the help text and make profile behaviour explicit per
  option.

### B36 — Seek and initial frame decode run on the UI thread
- **Where:** `src/main.rs` (`on_seek`, `load_clip`)
- **What:** `media::frame_at` / `frame_at_with` spawn ffmpeg and block until a
  full rgb24 frame returns, all on the Slint event-loop thread. The model
  preview correctly uses a worker; the before-frame path does not.
- **Impact:** Scrubbing the playhead freezes the whole UI for the duration of
  each decode (worse on long GOPs / slow disks). Opening a clip can hitch the
  window before the first paint settles.
- **Suggested fix:** Decode before-frames on the preview worker (or a small I/O
  worker), generation-guard the results, and keep the UI thread free.

### B37 — Preview deferred during export never restarts
- **Where:** `src/main.rs` (`request` closure; `spawn_exporter`)
- **What:** While `export_running` is true, `request()` bumps the preview
  generation and returns without queueing a job. When the exporter finishes it
  clears `export_running` but never re-invokes `request()`.
- **Impact:** After any export, the after-pane can stay blank/stale until the
  user wiggles a control or seeks. Loading a clip mid-export is especially
  easy to leave without a preview.
- **Suggested fix:** On export completion (success/cancel/fail), if a clip is
  still loaded, kick `request()` once; optionally queue a single deferred job
  instead of only bumping generation.

### B38 — Interpolated frame count falls back wrong when duration is missing
- **Where:** `src/export.rs` (`ExportSpec::output_frames`); `src/media.rs` (`probe`)
- **What:** RIFE output length is `(duration * target_fps).round()` only when
  `duration > 0`. If ffprobe reports no duration, the code falls back to
  `clip.frames` (source frame count), not
  `source_frames * target_fps / source_fps`.
- **Impact:** Containers without a reliable duration (some MKV/TS/raw pipes)
  export far too few frames at 48/60/120 fps, shortening the timeline and
  desyncing copied audio via `-shortest`.
- **Suggested fix:** When duration is unknown, derive output frames from
  `nb_frames` and the source/target rate ratio; keep duration-based math when
  duration is trusted.

### B39 — Export child cleanup can be skipped if pipe hand-off fails
- **Where:** `src/export.rs` (`run`, after both children spawn)
- **What:** `decoder.stdout.take()` / `encoder.stdin.take()` use `?` before the
  shared cleanup path. `Child` does not kill on drop, so a failed hand-off can
  leave ffmpeg processes running and stderr drain threads orphaned. The encoder
  spawn failure path does kill the decoder; this path does not.
- **Impact:** Rare, but a stuck decoder/encoder after a partial setup error
  burns CPU/GPU and can lock the output path.
- **Suggested fix:** Use a single RAII/cleanup guard from the moment either
  child exists; never `?` out without kill/wait/join.

---

### Low / polish

### B40 — Model output tensor size is not checked against native scale
- **Where:** `src/upscale.rs` (`run_tile` → `blend_into`)
- **What:** Inference output is blended assuming
  `len == 3 * (tile_w * scale) * (tile_h * scale)`. Wrong `models.json` scale,
  a bad probe, or a non-ESRGAN graph can under/over-read the buffer.
- **Impact:** Panic or silent corruption on mis-tagged drop-in models rather
  than a clear load-time error.
- **Suggested fix:** Validate output shape (or element count) after each
  `run_tile` / after `probe_scale`, and error with the model name and sizes.

### B41 — `.pth` loader reports CUDA from `is_available()`, not the session
- **Where:** `src/pth.rs` (`load_rrdb`)
- **What:** ONNX `.onnx` loading uses `error_on_failure` and an explicit CPU
  fallback. The `.pth` path registers CUDA without `error_on_failure` and labels
  the provider from `CUDA::is_available()` independently of what the session
  actually got.
- **Impact:** Edge installs can show/export as CUDA while ORT soft-failed to
  CPU, or hard-fail `.pth` loads that `.onnx` would have recovered from.
- **Suggested fix:** Mirror the `.onnx` provider policy: hard-require CUDA for
  this product, or fall back and label from the real EP list.

### B42 — Export progress total can disagree with frames actually written
- **Where:** `src/export.rs` (`output_frames` vs decode loop); `src/main.rs`
  (progress UI)
- **What:** Non-interpolated exports progress against `clip.frames` derived at
  probe time, while the decoder emits whatever the `fps=` filter produces.
  Rounding and VFR→CFR conversion can yield `done > total` or a bar that never
  reaches 100% before completion.
- **Impact:** Misleading ETA/status only; final success still depends on
  encoder exit, not the counter.
- **Suggested fix:** Prefer “frames written” without a brittle total, or
  recompute total after the first second of decode / from the fps filter policy.

---

## Completion checklist (prior audit — still closed)

Earlier B1–B32 items remain closed in code. They are not re-opened by this pass.

- [x] B1 — close exits and signals active export cancellation
- [x] B2 / B9 / B25 — export cancellation, child cleanup, and partial-file removal
- [x] B3 — both preview pan modes use a stable pointer-down origin
- [x] B4 — export dimensions are rounded to valid even values
- [x] B5 — decoder and encoder diagnostics are continuously drained
- [x] B6 — beside-app `models/` discovery no longer requires an index file
- [x] B7 / B8 / B20 — queue scheduling skips finished/running jobs and blocks duplicates
- [x] B10 / B11 / B13 / B18 — controls either run real processing or are honest and disabled
- [x] B12 — automatic tile size is real, persisted, and VRAM-aware
- [x] B14 — timing uses an explicit rational CFR policy
- [x] B15 — non-square-pixel sources are normalized to display geometry
- [x] B16 — inference discovers model input/output names
- [x] B17 — closed as not a bug; automated UI test confirms dock expansion
- [x] B19 — output paths are unique and the encoder refuses overwrite
- [x] B21 — unknown scale is labelled `auto`, then probed on load
- [x] B22 — machine-specific development runtime paths removed
- [x] B23 — closed as a non-bug after review
- [x] B24 — clean frame-boundary EOF and truncated-frame EOF are distinguished
- [x] B26 — Explorer file drops use the native window event stream and show hover feedback
- [x] B27 — missing GPU runtime no longer appears as an unexplained frozen preview *(launch pill still wrong — see B33)*
- [x] B28 — current video and export queue can be cleared independently
- [x] B29 — every selectable Source Restore filter runs in the bundled video engine
- [x] B30 — automatic tiles are capped safely for native 8x models
- [x] B31 — Frame Interpolation runs a genuine RIFE model in preview and export
- [x] B32 — Color Match affects RIFE and detail-model output in preview and export

### Open checklist (this audit) — all closed 2026-08-07

- [x] B33 — readiness derives from the CUDA payload; the probe reports device
      presence only
- [x] B34 — **closed as not a bug**, with a regression test. See below.
- [x] B35 — Source help corrected; Restore stays ungated by design
- [x] B36 — before-pane frames decode on a generation-guarded worker
- [x] B37 — preview resumes when an export releases the GPU
- [x] B38 — interpolated frame count scales by the rate ratio without duration
- [x] B39 — pipe hand-off failure tears down both children
- [x] B40 — tile output validated against the model's native scale
- [x] B41 — `.pth` mirrors the `.onnx` provider policy and labels what it got
- [x] B42 — progress stops quoting a total once writes exceed the estimate

#### B34 — closed as not a bug

**Filed as:** the lock estimates flow from `process_frame`'s `source`, which on
the interpolation path is a RIFE intermediate rather than an original decoded
pair.

**Why that is correct:** output N is produced from rendered frame N, whether
that frame was decoded or synthesized. The motion between consecutive *outputs*
is therefore the motion between consecutive *rendered* frames, which is what the
code uses. Substituting the original decoded pair would be the actual defect,
because interpolated outputs sit at intermediate times and would be warped by a
full source-interval of motion.

**Evidence:** `lock_with_interpolation_does_not_regress` exports the same clip
at 60 fps with Stability off and on. Flicker fell from 1.641 to 1.401
(**−14.6%**) — the combination improves output rather than smearing it.

The reasoning is also recorded as a comment at the lock site so it is not
"fixed" into the wrong behaviour later.

---

## Not bugs (known open work / by design)

| Item | Notes |
|------|--------|
| Preserve grain control | Explicitly labelled planned and `enabled: false` |
| IVTC / telecine | Designed; not implemented — do not treat as a silent defect unless UI claims it runs |
| Export codec/CQ UI | Footer is labelled fixed / LOCKED; values stay `h264_nvenc` / CQ 19 |
| Scale by target format (HD/4K) | Designed in DECISIONS §7 |
| SPAN / Compact / RealPLKSR `.pth` | RRDBNet only; unsupported topologies should error |
| LGPL ffmpeg → NVENC only, no libx264 | By design |
| AV1 NVENC on Ampere (3090) | Hardware limit, documented |
| Preview omits temporal lock | Intentional single-frame preview; export-only is disclosed in the Stability help |

**Doc drift note:** [NEXT-SESSION.md](NEXT-SESSION.md) still describes Stability as
disabled/planned in places, but the current code enables Stability by default
and runs `temporal::lock` on export. Treat the code as source of truth; refresh
session notes when convenient. That drift is not filed as a product bug.

---

## Already in good shape (for regression awareness)

- End-to-end export path: decode → optional RIFE → model → post → encode → audio map
- Cancel flag armed from UI / clear-queue / window close; partial outputs removed
- Unique output paths + ffmpeg `-n`
- Model preference ranking (Remacri default)
- Scale via resample-last after native model scale
- Settings persist model **by name**
- GPU telemetry polled off the UI thread
- Preview worker coalesces to newest job and generation-guards results
- SAR normalization and even geometry for yuv420p/NVENC
- SectionHeader TouchArea stacking (click-swallow trap fixed)
- Before/after divider tracks displayed image box

---

## Audit notes

- Method: static review of the current Rust/Slint tree only; no new runtime
  repro on GPU was required for the open items above (each has a direct code
  path demonstrating the fault).
- Do **not** treat this list as a substitute for DECISIONS open work; link both
  when prioritising.
- Priority suggestion if fixing later: **B33**, **B34**, **B36**, **B38**, then
  **B35** / **B37**.
- **Outcome:** all were addressed in one pass on 2026-08-07. One filed finding
  (B34) did not survive investigation — worth remembering that a static audit
  can describe behaviour accurately and still draw the wrong conclusion about
  whether it is a defect. Measure before changing something the audit calls
  broken.
