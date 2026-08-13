# Kraken Upscale — bug list

Static code audit against the current tree, informed by [DECISIONS.md](DECISIONS.md)
and [NEXT-SESSION.md](NEXT-SESSION.md).

**Scope:** defects and misleading behaviour only. Remaining planned product work
such as IVTC and grain preservation stays in DECISIONS / NEXT-SESSION unless UI
copy claims it is already active.

**Method:** full pass over `src/*.rs` and the export/preview control surface in
`ui/app.slint`, followed by regression tests and release-app UI verification.

Last updated: 2026-08-13

---

> **Status 2026-08-07:** every finding below is closed. B33 and B35–B42 were
> fixed; B34 was investigated and closed as not a defect. The descriptions are
> kept as the record of what was wrong and why.

### B55 — Row separators were drawn through action buttons

- **Where:** image/video Add-to-Batch row and collapsed bottom action bar
- **What:** one-pixel separators had a height but no explicit top coordinate,
  allowing Slint to center them vertically inside their parent rectangles.
- **Impact:** horizontal lines crossed button fills and labels, making active
  controls look struck through.
- **Fix:** affected separators are now explicitly anchored at `x = 0, y = 0`
  with full row width. The ready/transport row and collapsed action bar also
  have fixed, clipped layout bounds so their controls cannot overlap adjacent
  sections.
- **Verification:** live checks with an imported image confirm clean Add RTX to
  Batch and bottom action buttons with separators confined to row boundaries.
- [x] B55 — separators no longer cross buttons

### B54 — Output drawer opened automatically on a fresh launch

- **Where:** right-margin settings drawer at application startup
- **What:** the last saved right-side category was restored, and older/default
  settings selected a category instead of presenting a closed workspace.
- **Impact:** a fresh launch could immediately consume a large part of the
  workspace with OUTPUT controls the user had not requested.
- **Fix:** startup now always collapses the settings drawer. INPUT, ENHANCE,
  MOTION, and OUTPUT tabs toggle their own drawer open and closed. Upscale and
  Batch Upscale remain pinned along the bottom while the drawer is closed.
- **Verification:** fresh-launch visual check confirms only the right margin
  tabs are visible; opening and re-clicking OUTPUT expands and collapses it.
- [x] B54 — right settings drawer starts closed

### B53 — Resolution goals did not reveal whether they enlarge or shrink

- **Where:** OUTPUT resolution selector
- **What:** 720p, 1080p, 1440p, and 2160p were presented identically even when
  the loaded source was already larger than one or more of those targets.
- **Impact:** choosing a familiar label could silently request a downsize or an
  almost-no-op rather than an upscale.
- **Fix:** every resolution row and the selected value are now source-aware:
  red means downsize, amber means within 5% of the source short edge, green
  means a normal upscale, and cyan means a large upscale. A visible legend
  explains the colors. x2 remains green; x3/x4 are cyan.
- **Verification:** classification tests cover UHD→1080p downsize, exact and
  near-1080p source matches, normal and large named upscales, and multiplier
  classes.
- [x] B53 — resolution choices communicate downsize / same / upscale

### B52 — Output workflow exposed implementation details instead of user goals

- **Where:** right-side settings, resolution selection, task creation, and
  video/image output writers
- **What:** the UI only offered x2/x3/x4, kept every setting in one long rail,
  required an unnecessary Add Task step for one file, called the action
  Export, and locked output to one format.
- **Impact:** common 720p/1080p/1440p/2160p targets required mental arithmetic;
  the most common one-file workflow had an extra step; and codec, container,
  audio, and still-image format could not be selected.
- **Fix:** added aspect-preserving named goals alongside x2/x3/x4; split the
  settings into INPUT / ENHANCE / MOTION / OUTPUT vertical tabs on the app's
  far-right margin; pinned Upscale / Batch Upscale at the bottom; made Upscale
  automatically capture a loaded unqueued source; and added MP4/MKV,
  H.264/H.265 NVENC, CQ 15/19/23, audio copy/none, and PNG/JPEG/WebP choices.
  Named goals target output height for landscape and output width for portrait,
  with even video geometry and an exact aspect-preserving final resize.
- **Verification:** resolution math has portrait, landscape, multiplier,
  downscale-rejection, and >4x-rejection coverage. JPEG and lossless WebP are
  written and probed in the test suite. The release UI was visually checked
  with the right-margin menu on the extreme right and all seven resolution
  choices present. A clean single-image release run with an empty batch turned
  one click on Upscale into a completed 256×256 → 512×512 CUDA task without
  first using Add to Batch.
- [x] B52 — goal-based sizing, margin menus, direct Upscale, and formats

### B51 — Activity log described progress but not the pipeline

- **Where:** video previews, all three image preview paths, video/image/RTX VSR/
  Generative exports, and failure reporting
- **What:** the log showed generic model-load messages and tile counters but did
  not say which source-preparation, GPU, temporal, finishing, or output stage
  was running. Export queue progress was visible in the row but mostly absent
  from the persistent activity log.
- **Impact:** a healthy long operation could look opaque, and diagnosing a stall
  or bad result required guessing what happened before and after inference.
- **Fix:** added named START/STEP/PROCESSING/FINALIZING/COMPLETE/FAILED entries.
  The trace now records source format and treatment, decode and model-load
  timing, exact dimensions, model file/precision/provider, temporal window,
  tile/overlap/stride counts, enabled RIFE/color/limit/blend/stability stages,
  codec/CQ/audio behavior, useful milestone ETAs, generative pre-upscale and
  refine phases, output path/byte size, and precise cancellation/error context.
  Long video exports log percentage milestones instead of every frame.
- **Verification:** compile, all 40 normal tests, strict warning checks, release
  build, and a real release-app trace using `SpaceVoyage_2.mp4`. The trace
  covered decode/RIFE source preparation, model file and 0.03-second CUDA load,
  the genuine five-frame window, all 15 effective GPU tiles with ETA, color and
  exact-resize finishing, and successful completion in 4.6 seconds.
- [x] B51 — activity log provides a readable step-by-step pipeline trace

### B50 — The good RTX VSR image path existed only in ComfyUI

- **Where:** Images mode, preview/export backends, optional runtime packaging
- **What:** the tested `RTX-VSR-PRO.json` workflow produced a strong fast 2×
  result through NVIDIA Video Super Resolution at ULTRA quality, but Kraken
  offered only conventional ONNX models or the much slower FLUX refine path.
- **Impact:** the most practical faithful result required leaving Kraken and
  running a separate application. Users could also mistake Generative Refine
  for the equivalent option even though it has completely different speed and
  detail-invention behavior.
- **Fix:** added **RTX VSR** as a third Images pipeline with x2/x3/x4 and
  Low/Medium/High/Ultra choices. Kraken calls NVIDIA's Video Effects SDK DLLs
  directly from a colocated `rtx-vsr/` folder; it does not launch ComfyUI or
  require an external scripting runtime. Preview and queue exports use the
  same backend, write lossless PNG, drop idle ONNX/RIFE sessions before export,
  and report native CUDA failures instead of falling back. Outputs above the
  verified 15,360px VSR edge use the Pro workflow's protected intermediate and
  an exact final resize; larger than 16,384px is rejected. A channel-collapse
  guard prevents the known 16K corrupted result from being saved.
- **Verification:** the native hardware smoke loaded the official 0.1.0.1
  runtime on the RTX 3090, processed 640×360 RGB through ULTRA, returned a
  non-empty 1280×720 frame, and released the effect and GPU buffers. The final
  release UI also processed the supplied 720×1280 source to 1440×2560 through
  Ultra in 1.0 seconds on CUDA. All 40 normal tests pass; the hardware test
  also passes when explicitly enabled.
- [x] B50 — native NVIDIA RTX VSR is a real selectable Images pipeline

### B49 — Generative export looked frozen and used an unsuitable FP32 model

- **Where:** Generative Refine package, native request progress, queue status
- **What:** the first deployed FLUX transformer was a 23.8 GB FP32 file. On the
  24 GB RTX 3090 it left no useful workspace, so the native runtime streamed
  model layers through system memory. The queue only updated after a whole tile
  returned, hiding all model loading and diffusion-step activity.
- **Impact:** fans ran while GPU telemetry stayed low, `tile 1/15` appeared
  frozen, and a measured diffusion step took about 161 seconds. The projected
  export was well over an hour.
- **Fix:** replaced FP32 with the pinned off-the-shelf FLUX.1 Dev Q6_K GGUF,
  reject >18 GiB diffusion files on 24 GB-class cards, and poll the native log
  during each request. Queue status now reports tile, real diffusion step,
  elapsed time, ETA, and model-preparation time. Tensor-loading/VAE counters are
  excluded so progress cannot jump backward.
- **Verification:** exact Q6 SHA-256 verified; a real 832×832 CUDA img2img tile
  at the UI's 15-step/0.10 settings completed in 12.8 seconds versus a roughly
  322-second FP32 projection from its measured 160.77 seconds/iteration—about
  25× faster. Step-parser regression and strict lint pass.
- [x] B49 — responsive generative progress and a 3090-appropriate FLUX package

### B48 — No manual way to release Kraken's idle GPU allocations

- **Where:** System panel and long-lived export worker
- **What:** the export worker deliberately retained the last upscale/RIFE
  sessions between queued jobs for speed, but there was no user control to
  drop those caches when another GPU-heavy application needed the card.
- **Impact:** the System panel could show nearly all VRAM occupied after work
  had stopped, with no recovery path short of closing Kraken.
- **Fix:** added **Release GPU Memory** beneath the live GPU telemetry. It
  invalidates superseded preview delivery and tells the owning export worker to
  drop its idle upscale and interpolation sessions on their correct thread.
  During an active export it becomes **Cancel + Release GPU**, requests clean
  cancellation, stops the native model process, then clears cached sessions.
  It explicitly notes that it cannot reclaim memory owned by another
  application or Kraken instance.
- **Verification:** command wiring compiles through the Slint callback, cache
  ownership remains on the worker thread, and the release path logs completion
  only after both cached sessions have been dropped.
- [x] B48 — manual idle GPU/cache release control

### B47 — Images mode lacked the checkpoint-driven refine path from Kraken Art

- **Where:** Images mode, export task capture, CUDA runtime packaging
- **What:** the conventional still-image workflow stopped after its ONNX image
  model. It could not reproduce Kraken Art's Ultimate-style sequence of a
  faithful pre-upscale followed by low-denoise FLUX img2img tiles and an exact
  resolution snap.
- **Impact:** Images mode could enlarge and sharpen, but it could not use a
  diffusion checkpoint to reconstruct texture the way the grandfather app or
  an Ultimate SD Upscale workflow can.
- **Fix:** added a separate Generative Refine pipeline. It runs the selected
  conventional model first, releases that CUDA session, starts a bundled
  native `stable-diffusion.cpp` CUDA server on loopback, refines padded FLUX
  tiles, blends overlaps with raised-cosine weights, restores source color and
  alpha, applies an optional restrained finish, snaps to the requested size,
  and shuts the server down. Complete downloadable model packages are declared
  by JSON under `models/generative/`; no model conversion or external runtime
  is required.
- **Verification:** native runtime device discovery identified the RTX 3090;
  manifest/package discovery, odd-canvas tile coverage, PNG transport,
  deterministic spatial seeds, the complete test suite, and strict lint all
  pass. After the prior app instance was closed, the release integration test
  loaded the complete FLUX.1 Dev package and generated a real 512×512 img2img
  tile on CUDA. The native server exited cleanly and GPU use returned to about
  0.9 GB used / 23.4 GB free.
- [x] B47 — native Ultimate-style generative still-image pipeline is wired and packaged

### B46 — Replacing an image could temporarily compare two different sources

- **Where:** image open/drop handling and asynchronous preview delivery
- **What:** opening a second still updated the source metadata immediately, but
  left the previous before/after panes visible until the new CUDA preview was
  ready. A previous worker result that had already queued its UI callback could
  also arrive after the replacement token changed.
- **Impact:** Split view could show the new source on one side and the old
  image's upscale on the other, implying that Kraken had combined two images.
- **Fix:** a successful image replacement now invalidates pending preview work
  and clears both panes before loading the new source. Source, progress, and
  completed-preview callbacks re-check their generation on the UI thread, at
  the moment they mutate the panes.
- **Verification:** replacement-token regression coverage, complete release
  suite, and strict lint pass.
- [x] B46 — image replacement is atomic and stale preview delivery is rejected

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

### B44 — Recommended video-compatible models were absent from the picker

- **Where:** model deployment, discovery, and Detail Model UI
- **What:** only the genuine NanoVSR multi-frame graph and the older still-image
  ESRGAN collection were installed. The compact Real-ESRGAN video variants,
  Real-CUGAN, and representative SPAN/Compact models discussed for fast video
  work were not actually present.
- **Impact:** the UI gave no fast, conservative video-tuned choices and made it
  look as though the recommendations had been implemented when they had not.
- **Fix:** added a curated `models/video-frame/` collection with Real-ESRGAN
  General x4v3, Real-ESRGAN AnimeVideo v3, two Real-CUGAN strengths, AniSD
  General/Cel SPAN, OpenProteus Compact, and ModernSpanimation V1. The picker
  now labels genuine temporal models `VIDEO · multi-frame`, this collection
  `VIDEO · fast frame model`, and the older collection `IMAGE`.
- **Verification:** all eight new graphs loaded through the CUDA provider and
  completed native-scale inference. The incompatible ModernSpanimation FP16
  graph was deliberately excluded instead of being silently selected.
- [x] B44 — video-compatible Real-ESRGAN / Real-CUGAN / SPAN set installed,
      distinguished honestly, and GPU-verified

### B45 — Still-image upscalers had no image workflow

- **Where:** application mode, import/drop handling, preview, model picker,
  queue, and export
- **What:** the original ESRGAN/RRDB models were retained as "image" models,
  but Kraken could only open a video and encode MP4. There was no way to use
  those models on a PNG, JPEG, WebP, BMP, or TIFF.
- **Impact:** separating video-tuned and still-image models made the taxonomy
  honest but stranded the entire still-image collection.
- **Fix:** added a persistent Image mode with image-aware browse and automatic
  drag/drop routing, image-only model filtering, still preview, mixed queue
  support, independent clearing, and lossless PNG export. Source alpha is
  resized separately and restored after RGB inference, so transparent artwork
  remains transparent. Video-only Source, Stability, and RIFE controls are
  absent in Image mode.
- **Verification:** a real CUDA export transformed the transparent 256×256
  Kraken artwork into a 512×512 PNG with its alpha channel intact. Image
  probing/decoding, extension routing, model-list separation, settings
  migration, and mode persistence have regression coverage.
- [x] B45 — functional conventional Image mode using the existing image models

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
