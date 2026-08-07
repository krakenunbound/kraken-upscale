# Start here next session

Read [DECISIONS.md](DECISIONS.md) for architectural reasoning and
[bugslist.md](bugslist.md) for the completed reliability audit.

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
- Planned Source, Stability, Color, and Interpolation controls are clearly
  disabled instead of implying an active pipeline.
- The slide-out dock is covered by an automated width/animation test.

## Next product work

1. **Temporal lock** — flow-guided stabilization of generated detail. This is
   still the main feature required to close the video-quality gap.
2. **Source restoration** — deinterlace/IVTC, deblock, grain separation, and
   re-injection for older TV/DVD material.
3. **Performance** — tile batching, larger/whole-frame execution where memory
   permits, and optional TensorRT caching.
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
