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

It was verified from that exact path with the root `cuda/`, `models/`, provider
DLLs, and bundled video tools. These generated payloads are intentionally
ignored by Git and should be refreshed after a source rebuild.

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

1. **Temporal lock** — flow-guided stabilization of generated detail. This is
   still the main feature required to close the video-quality gap.
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
