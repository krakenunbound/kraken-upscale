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

The video picker also includes a GPU-verified fast set under
`models/video-frame/`: Real-ESRGAN General x4v3 and AnimeVideo v3, two
Real-CUGAN strengths, AniSD General/Cel SPAN, OpenProteus Compact, and
ModernSpanimation V1. These remain single-frame graphs and are labelled that
way. `models/video/` contains genuine neighboring-frame models such as NanoVSR.

The toolbar now has a separate Images mode. It accepts PNG/JPEG/WebP/BMP/TIFF,
filters the picker to the existing still-image models, previews through the
native tiled CUDA path, and writes lossless PNG tasks. Alpha is preserved
separately from RGB inference. Video and image tasks may coexist in the queue,
and each mode remembers its own selected model.

Images mode now also has a **Generative Refine** submode. It runs the selected
faithful image model first, drops that CUDA session, then uses a bundled native
`stable-diffusion.cpp` CUDA server for low-denoise FLUX img2img tiles. Complete
model bundles are JSON manifests under `models/generative/`; the currently
staged FLUX.1 Dev package reuses the already-downloaded weights through NTFS
hardlinks, so it consumes no second 28 GB copy. This remains export-only because
a diffusion preview would be slow and would misleadingly reserve most VRAM.

Both modern ZIP checkpoints and the legacy stream format used by many community
models are supported. Old ESRGAN and newer RealESRGAN RRDBNet key layouts are
recognized, including 2× input rearrangement and 2×/4×/8× native scales.

The System panel now includes **Release GPU Memory**. It drops the export
worker's idle upscale/RIFE sessions without closing Kraken. Preview sessions
are already job-local; an active preview finishes and releases itself. The
button is disabled during export and cannot affect VRAM owned by another
process.

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
models/video-frame/              fast video-tuned frame models
models/video/                    genuine neighboring-frame models
models/interpolation/            RIFE timestep model
models/generative/               manifests + complete generative model packages
native-diffusion/                bundled stable-diffusion.cpp CUDA runtime
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

All findings through B46 are closed — see [bugslist.md](bugslist.md) and
[CHANGELOG.md](CHANGELOG.md). B34 was closed as not a defect after measurement
contradicted it. B43 added genuine multi-frame inference; B44 installed and
clearly classified the recommended fast video-compatible model families. B45
added the separate still-image workflow for the existing image models.
B46 made replacement atomic so a superseded preview cannot be shown beside a
newly dropped image.

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
- Grain preservation remains clearly labelled and disabled instead of implying
  an active pipeline. Stability is now real — see below.
- The slide-out dock is covered by an automated width/animation test.
- Inverse telecine reverses 3:2 pulldown for film-sourced broadcast video, with
  field detection reading pixels via `idet` rather than trusting container
  metadata, which a remux silently drops.

## Target content — read this before prioritising

Audited 2026-08-07 against the real library. **There are no DVDs**; ~99% of use
is downloaded rips:

```text
1440×1080 or 1920×1080 · H.265 Main 10 · yuv420p10le · progressive · 23.976
```

The SD/broadcast assumptions in earlier notes are wrong. IVTC and deinterlace
are built and tested but are insurance, not the main path — they will rarely
fire. See DECISIONS §1.

**The pipeline currently decodes to 8-bit `rgb24`**, so 10-bit sources are
quantised before the model sees them and encoded back at 8 bits. That is the
largest quality gap for this material, and it shows as banding in gradients —
space, nebulae, dim interiors.

A 10-second real sample is at
`F:\MiniMaxH3\outputs\2026-08-06\voyager-10s.mkv` (339 frames, 6 MB).

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
2. **Genuine video super-resolution — first path shipped.**
   - `models/video/NanoVSR-644k-T5.onnx` consumes five neighboring frames and
     returns a five-frame 4x sequence; it is not a renamed image model.
   - Export uses bounded windows and spatial overlap-save tiling. Final partial
     windows repeat the last frame and discard padding; hard cuts start a new
     window so recurrent detail cannot leak between shots.
   - Preview decodes a real temporal neighborhood. The legacy Stability lock is
     bypassed for video models and remains available for frame models.
   - Next comparison: RealBasicVSR / BasicVSR++ quality versus this portable,
     lightweight baseline. Do not count SPAN, CUGAN, DAT2 or RealPLKSR as video
     models; they remain frame-model/performance work.
3. **10-bit pipeline** — **highest value for the actual library.** Decode
   `rgb48`, keep precision through the model (which already works in f32/f16),
   encode 10-bit HEVC. Removes both the input quantisation and the 8-bit output
   bottleneck.
4. **Source restoration** — deinterlace, deblock and **inverse telecine** are
   active and verified against ground truth. Grain separation and re-injection
   remain, and matter more than expected for film-sourced Blu-rays.
5. **Performance** — measured larger tiles and parallel post-processing landed;
   tile batching and optional TensorRT caching remain.
6. **Broader frame-model families** — ONNX SPAN, Compact, Real-CUGAN, and compact
   Real-ESRGAN choices are now installed and verified. Direct checkpoint import
   still supports RRDBNet only; RealPLKSR and additional topology builders
   remain optional future work.
7. **Target-format scaling** — HD/2K/4K presets remain designed but unbuilt.
8. **Generative image refinement** — **implemented** as the separate Images
   submode using a complete checkpoint/text-encoder/VAE package and bundled
   native CUDA runtime. Kraken Art's pre-upscale → img2img tiles → exact snap
   contract is ported. Padded raised-cosine overlap replaces simple feather
   paste. A second seam-repair pass and shared global latent noise remain
   quality experiments, not requirements for the shipped first pass.

## Regression assets

```text
F:\MiniMaxH3\outputs\2026-08-06\SpaceVoyage.mp4
F:\MiniMaxH3\outputs\2026-08-06\SpaceVoyage_2.mp4
```

`SpaceVoyage_2x_4x-UltraSharp.mp4` beside those inputs is the known-good export
reference: 1408×2496, 120 frames, 24 fps, and intact audio.
