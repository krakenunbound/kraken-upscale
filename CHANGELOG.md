# Changelog

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
- Changed Export Current to Export Next so the label matches queue behavior.
- Added `-shortest` to keep remuxed audio/video duration bounded.

### Media correctness

- Probed both average and nominal frame rates and retained the selected rational
  rate throughout decode and encode.
- Added explicit conversion to constant frame rate for the raw-frame pipeline.
- Probed sample/display aspect and normalized anamorphic sources to even,
  square-pixel display dimensions before inference.
- Applied the same geometry policy to preview frames and exports.

### Interface and lifecycle

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
- Marked Source, Stability, Color, and Frame Interpolation as planned and
  disabled their controls until their pipelines exist.
- Made the fixed export-format footer visibly non-interactive.
- Changed unknown model scale display from `0×` to `auto`.

### Verification

- Added tests for odd-dimension output sizing, exact EOF, truncated frame input,
  anamorphic media normalization, frame decoding, and dock width animation.
- Verified direct import and inference with unchanged RealESRGAN 2×/4×,
  UltraSharp 4×, Remacri 4×, and NMKD Faces 8× checkpoints.
- Confirmed the complete test suite and release build pass.
