# Start here next session

Read [DECISIONS.md](DECISIONS.md) for architectural reasoning and
[bugslist.md](bugslist.md) for the completed reliability audit. Read
[RIFE-IMPLEMENTATION.md](RIFE-IMPLEMENTATION.md) before changing interpolation,
FPS scheduling, cut detection, or color matching.

## Current state

The app is a self-contained native Windows executable. Users place standard
RRDBNet `.pth` checkpoints or `.onnx` models in the beside-app `models/` folder;
the app discovers and loads them directly. `models.json` is optional metadata.

Direct import and inference are verified with unchanged community files:

- RealESRGAN 2× and 4×
- UltraSharp 4×
- Remacri 4×
- NMKD Faces 8×

Both modern ZIP checkpoints and the legacy stream format used by many community
models are supported. Old ESRGAN and newer RealESRGAN RRDBNet key layouts are
recognized, including 2× input rearrangement and 2×/4×/8× native scales.

## Build and run

```text
cargo build --release
target\release\kraken-upscale.exe "path\to\clip.mp4"
```

A path argument is optional. The app also accepts window drag-and-drop and the
Import button.

The current easy-launch release is also staged at:

```text
F:\Video Upscale\Kraken Upscale.exe
```

The staged folder is **relocatable**: every payload resolves relative to the
executable.

```text
Kraken Upscale.exe
ffmpeg/                          ffmpeg + ffprobe and their DLLs
cuda/                            CUDA 13 + cuDNN 9 redistributables
models/                          upscale models (.onnx / .pth)
models/interpolation/            RIFE timestep model
onnxruntime_providers_cuda.dll
onnxruntime_providers_shared.dll
```

`ffmpeg/` beside the exe is required, not optional: without it `media::bin_dir`
falls back to a compile-time absolute source path and the build only runs from
the machine it was built on.

To refresh after a rebuild:

```text
copy target\release\kraken-upscale.exe "Kraken Upscale.exe"
copy target\release\onnxruntime_providers_*.dll .
robocopy vendor\ffmpeg\bin ffmpeg /XF ffplay.exe
```

These generated payloads are intentionally ignored by Git.

## Bug audit

All ten findings from the 2026-08-07 audit (B33–B42) are closed — see
[bugslist.md](bugslist.md) and [CHANGELOG.md](CHANGELOG.md). Nine were fixed;
**B34 was closed as not a defect** after measurement contradicted it. Worth
remembering: a static audit can describe behaviour accurately and still reach
the wrong conclusion about whether it is broken.

## Completed reliability pass

- Window close exits and requests active export cancellation.
- Export cancellation kills and waits for both media children and removes the
  partial output.
- Decoder truncation is distinguished from a clean frame-boundary EOF.
- Encoder/decoder diagnostics are drained continuously.
- Odd output dimensions are normalized for the encoder.
- Duplicate queue scheduling and silent output overwrite are prevented.
- Non-square-pixel sources are normalized to square-pixel display geometry.
- Timing uses an explicit rational CFR conversion policy.
- Preview panning is stable in both layouts.
- Explorer file drag-and-drop uses the native window event stream and provides
  visible hover feedback over the empty preview stage.
- Automatic tile sizing is persisted and based on available VRAM.
- The RTX 3090 profile uses the measured 896px optimum (2.56s versus 4.57s at
  512px on `SpaceVoyage_2.mp4`), native 8x models receive a safe scale-aware
  cap, and CPU post-processing is parallelized.
- Source profile deinterlacing/restoration and source-color matching are active
  in both preview and export.
- Genuine RIFE interpolation is active at 48/60/120 fps through a drop-in ONNX
  timestep model, with hard-cut protection and color correction in preview and
  export.
- Clear Video and Clear Queue are available independently; new previews defer
  while an export owns the GPU.
- Stability and grain preservation remain clearly labelled and disabled
  instead of implying an active pipeline.
- The slide-out dock is covered by an automated width/animation test.

## Next product work

1. **Temporal lock** — **shipped and enabled**, but not yet signed off.
   - Phase 0 baseline instrumentation: `src/stages.rs`, per-stage timings in
     the activity log. Steady state 1.65 s/frame, model 98% of it.
   - `src/temporal.rs`: built-in pyramid block matching, confidence gate
     (match residual × forward–backward consistency), scale-aware warp and
     confidence-weighted blend. Runs in export; Stability is a real control.
   - Measured: **+154.6% of model-added flicker removed** at **+2.2% wall
     cost** on near-static content, against a ≥50% bar.
   - Confidence releases with displacement (16 px, set from measurement), and
     **the preview now shows the lock** — scrub off frame zero with Stability
     on and the status line reports `lock 0.75 × held 0.80`.
   - **No-cut-ghosting is verified** on `assets/hardcut-test.mp4`, a spliced
     clip with a real edit: one cut found, confidence zeroed, locked output
     byte-identical to fresh detail.
   - **One thing remains, and it needs your eyes, not more measurement:**
     **pick the default Lock value.** 0.75 measurably over-locks on near-static
     content — locked output falls below the no-invented-detail floor, so real
     source change is being smoothed too. That is inherent to a temporal filter
     (DECISIONS §12) and cannot be metric-tuned away: optimising the residual
     drives the setting to 1.0, which is wrong. Compare 0.5 / 0.65 / 0.75 in
     the preview — a minute's work now that the lock is previewable.

   Reproduce the measurements:

   ```text
   set KRAKEN_CUDA_DIR=F:\Video Upscale\cuda
   set KRAKEN_TEST_CLIP=F:\MiniMaxH3\outputs\2026-08-06\SpaceVoyage_2.mp4
   cargo test --release locked_export_reduces_flicker -- --ignored --nocapture
   cargo test --release flow_tracks_real_frames      -- --ignored --nocapture
   cargo test --release baseline_stage_timings       -- --ignored --nocapture
   ```

   GPU tests need the ORT provider DLLs copied from `target/release/` into
   `target/release/deps/`, or they silently fall back to CPU and fail an
   unrelated assertion.
2. **Source restoration** — deinterlace and deblock are active; IVTC and grain
   separation/re-injection remain.
3. **Performance** — measured larger tiles and parallel post-processing landed;
   tile batching and optional TensorRT caching remain.
4. **Broader model families** — SPAN, Compact, and RealPLKSR need their own
   native topology builders; current direct checkpoint support is RRDBNet.
5. **Target-format scaling** — HD/2K/4K presets remain designed but unbuilt.

## Regression assets

```text
F:\MiniMaxH3\outputs\2026-08-06\SpaceVoyage.mp4
F:\MiniMaxH3\outputs\2026-08-06\SpaceVoyage_2.mp4
```

`SpaceVoyage_2x_4x-UltraSharp.mp4` beside those inputs is the known-good export
reference: 1408×2496, 120 frames, 24 fps, and intact audio.
