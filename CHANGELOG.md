# Changelog

## Unreleased — Phase 1 step 2: confidence gate

### Trust, then blend

- Added `temporal::analyze`, which estimates motion and decides per grid cell
  how far it can be trusted, and `temporal::lock`, which blends the warped
  previous output into current detail in proportion to that trust. Neither is
  wired into the pipeline yet.
- Two independent checks, chosen because they fail on different things:
  - **match residual** catches brightness change, where a fade leaves no good
    match at any offset;
  - **forward–backward consistency** catches occlusion and motion boundaries,
    which residual can miss when a wrong match happens to score well.
- They are **multiplied, not averaged**. Either failing alone makes the warp
  untrustworthy, and averaging would let a good residual mask an occlusion.
- Scene cuts disable the lock outright rather than softening it, so no invented
  detail is carried across an edit.

### Validated against the previously measured failure modes

The gate found both without being told about them:

| pair | residual reduction | confidence |
|---|---|---|
| 167→168 (9.8 px motion) | +17% — worst | **0.34** — lowest |
| 0→1 (brightness change) | +2% — worst | **0.45** |
| good pairs | +30…+53% | 0.79–0.84 |

- Cost rises from 5–8 ms to **13–15 ms** per pair, since flow now runs in both
  directions. Still under 1% of the 1.65 s/frame model budget.

## Unreleased — Phase 1 step 1: optical flow

### Temporal lock groundwork

- Added `src/temporal.rs`: motion estimation, warping, and the mean-absolute-
  difference metric the lock will be measured by. Not yet wired into the
  pipeline.
- Resolved ENHANCEMENTS decision checkpoint 1 in favour of a **built-in
  estimator** rather than a downloaded RAFT ONNX. Reasoning is in the module
  header: the lock is not optional the way interpolation is, so requiring a
  second download would mean most users never get it. The confidence gate, not
  flow accuracy, is what keeps poor estimates safe.
- Coarse-to-fine block matching on a luma pyramid, with a median filter over
  the vector field and a small zero-bias so flat regions resolve to no motion
  rather than noise.

### Measured

- Flow costs **5–8 ms per frame pair** — 0.4% of the 1.65 s/frame model budget.
- Residual falls on every pair measured and never rises.
- On `SpaceVoyage_4` (240 frames, real motion), sampling 10 pairs spread across
  the clip: **mean residual reduction +35.3%**, best +58%.
- `SpaceVoyage_2` is nearly static (0.02–0.76 px mean motion) and exercises only
  the flicker case; the measurement test now samples across the whole clip
  rather than taking six consecutive frames from the opening shot.

### Identified for the confidence gate

Two failure modes, both to be handled by declining to lock rather than by
improving the estimator:

- **Brightness change** — SAD is not intensity-invariant, so a fade or lighting
  shift leaves no good match. The clip's first frame pair reduced residual by
  only 2% despite 3.6 px of apparent motion.
- **Large or discontinuous motion** — at 9.8 px mean motion the reduction fell
  to 17%. Block matching is weakest at motion boundaries, and the median filter
  that removes outliers also smooths across genuine edges.

### Fixed

- **`frame_at` returned the wrong frame.** The seek time was formatted to
  millisecond precision, but a 24 fps frame spans 41.667 ms — so frame 2 begins
  at 0.083333 s while the request rounded to `0.083`, landing back inside frame
  1. Timeline scrubbing could show a stale frame. Found because two of six
  measured frame pairs reported exactly 0.00 residual while the same frames
  extracted sequentially were all distinct.

## Unreleased — Phase 0 export instrumentation

### Stage timing

- Added `src/stages.rs`: per-stage accumulator for the export pipeline
  (decode, interpolate, model, post, encode).
- Changed `export::run` to return the timing breakdown; the summary is written
  to the activity log when an export completes.
- Separated the first model call from the steady-state average. It carries
  one-off CUDA session creation and cuDNN autotuning that never recur, and
  folding it into the mean makes short runs look far slower than reality.
- Added `baseline_stage_timings`, an ignored test that measures an unlocked
  reference export and prints the breakdown, giving the temporal lock something
  concrete to be judged against.

### Measurement corrections

- **The previously recorded 2.56 s "model pass @ 896px tiles" is the one-off
  warm-up cost, not the per-frame model time.** Measured steady state on the
  same clip and tile size is **1.65 s/frame** — 36% faster than documented.
- The model stage is **98% of measured time**. Decode (5 ms), post (32 ms) and
  encode write (1 ms) per frame are negligible, so optimisation effort belongs
  entirely in inference.
- A first run immediately after a build measured 17.1 s/frame versus 1.77 s on
  the next identical run. Cold-start variance is roughly 10×, so any benchmark
  must discard its first run.

### Test environment

- ONNX Runtime provider DLLs are emitted to `target/release/` but GPU tests run
  from `target/release/deps/`. They must be copied across, and `KRAKEN_CUDA_DIR`
  must point at the repository `cuda/` folder, or tests silently fall back to
  CPU and fail an unrelated assertion.

## Unreleased — 2026-08-06 reliability and native-model pass

### Model loading

- Added direct `.pth` checkpoint loading in the Windows app.
- Added readers for both ZIP checkpoints and legacy PyTorch stream checkpoints.
- Added in-memory RRDBNet graph construction for old ESRGAN and new RealESRGAN
  state-dictionary layouts.
- Added native-scale inference for 2×, 4×, and 8× models.
- Added RealESRGAN 2× space-to-depth input handling.
- Added dynamic model input/output name discovery.
- Made the beside-app `models/` folder authoritative without requiring
  `models.json`; optional metadata remains supported.
- Removed the obsolete external model-preparation utility.

### Export reliability

- Added user cancellation for the active export batch.
- Guaranteed decoder and encoder kill/wait cleanup on cancel and all error paths.
- Removed partial output files after failed or cancelled exports.
- Continuously drained decoder and encoder diagnostics to prevent pipe stalls.
- Distinguished clean frame-boundary EOF from a truncated raw frame.
- Rounded output dimensions to valid even values for 4:2:0 encoding.
- Added no-overwrite encoder behavior and unique queued output paths.
- Prevented finished, running, or already-scheduled jobs from being queued twice.
- Changed queue scheduling to select the next unfinished task; the final primary
  button label is `Export`.
- Added `-shortest` to keep remuxed audio/video duration bounded.

### Media correctness

- Restored functional Source profiles: broadcast profiles can deinterlace
  detected fields, and the Restore selector applies light/strong deblocking or
  heavy-compression cleanup before inference.
- Kept every Restore option within the filters shipped by the bundled video
  engine and added execution coverage for all four selector values.
- Added real source-color matching with bounded per-channel mean/contrast
  correction in preview and export.
- Added a genuine RIFE timestep stage using a drop-in MIT-licensed ONNX model.
  It synthesizes motion-aware frames before detail upscaling at 48, 60, or
  120 fps, skips interpolation across detected hard cuts, and color-matches
  generated frames to their source-frame blend when Color Match is enabled.
- Added [RIFE-IMPLEMENTATION.md](RIFE-IMPLEMENTATION.md) with the permanent
  model provenance, pipeline order, scheduling, logging, performance,
  verification, and limitations record.
- Probed both average and nominal frame rates and retained the selected rational
  rate throughout decode and encode.
- Added explicit conversion to constant frame rate for the raw-frame pipeline.
- Probed sample/display aspect and normalized anamorphic sources to even,
  square-pixel display dimensions before inference.
- Applied the same geometry policy to preview frames and exports.

### Interface and lifecycle

- Added Clear Video and Clear Queue actions; clearing a running queue also
  requests cancellation, while clearing a video leaves queued snapshots intact.
- Restored the primary button label from `Export Next` to `Export`.
- Deferred new-video previews during an active export so the next video can be
  configured without competing for GPU time.
- Made missing GPU runtime a visible blocking state instead of silently starting
  an impractically slow CPU preview.
- Added preview start, model-load, per-tile progress/ETA, completion, and failure
  entries to the activity log.
- Mirrored activity entries to a persistent `activity.log` beside the saved
  settings so a failed or stalled session remains diagnosable after restart.
- Normalized unavailable sample-aspect metadata without showing a false warning.
- Replaced the conflicting legacy file-drop hook with the window framework's
  native Explorer drop events and added whole-stage hover feedback.
- Corrected compounding preview pan movement in split and side-by-side modes.
- Closing the main window now exits the event loop after saving settings and
  requesting active export cancellation.
- Replaced the nonfunctional hardware selector with an honest automatic-backend
  status.
- Replaced the inaccurate half-precision performance promise with neutral copy.
- Made automatic tile sizing real, persisted, and VRAM-aware.
- Kept Stability and grain preservation visibly planned and disabled until
  their pipelines exist.
- Activated the Frame Interpolation controls only when the RIFE model is
  present, and included target FPS/frame count in queued task details and logs.
- Made the fixed export-format footer visibly non-interactive.
- Changed unknown model scale display from `0×` to `auto`.

### Root launch build

- Built the optimized Windows executable and staged an easy-launch copy at
  `F:\Video Upscale\Kraken Upscale.exe`.
- Staged the CUDA provider DLLs and redistributable CUDA runtime beside the root
  executable; the existing root `models/` and `vendor/ffmpeg/` folders complete
  the local launch layout.
- Launched that exact root copy with `SpaceVoyage_2.mp4` and verified CUDA model
  loading, color-matched preview completion, model discovery, and a responsive
  window.
- Root executable SHA-256:
  `5ACD0B75C04CAC587A4594F5810B954B093B96799E6E25B939B05F1535D1B306`.
- Generated executable/runtime payloads remain outside Git; source, build
  instructions, model provenance, and change records stay committed.

### Verification

- Benchmarked automatic tile sizes on the target RTX 3090 using
  `SpaceVoyage_2.mp4`: 896px reduced the model pass from 4.57s to 2.56s versus
  512px.
- Raised the 24 GB automatic tile profile to the measured 896px optimum and
  parallelized color, blend, clamp, and resampling work across CPU cores.
- Made tiles scale-aware so native 8x models stay within a safe 4096px output
  pitch while 2x/4x models still benefit from larger tiles.
- Verified a real arbitrary-timestep 24→60 fps export through RIFE, color
  matching, detail upscaling, and NVENC, including output FPS and frame count.
- Measured full-size RIFE inference plus color correction at 0.31 seconds for a
  704×1248 frame pair on the target RTX 3090.
- Added tests for odd-dimension output sizing, exact EOF, truncated frame input,
  anamorphic media normalization, frame decoding, and dock width animation.
- Verified direct import and inference with unchanged RealESRGAN 2×/4×,
  UltraSharp 4×, Remacri 4×, and NMKD Faces 8× checkpoints.
- Confirmed the complete test suite and release build pass.
