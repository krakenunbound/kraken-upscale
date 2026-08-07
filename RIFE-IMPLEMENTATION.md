# RIFE interpolation and color pipeline

## What changed

Kraken now performs genuine motion-aware frame interpolation in preview and
export. The old settings-only control was replaced by a native RIFE ONNX stage
with selectable 48, 60, and 120 fps targets.

Color Match is also active in both stages that can introduce color drift:

1. A RIFE-generated frame is matched to the corresponding linear color
   reference between its two source frames.
2. The detail-model result is matched back to the frame that entered the
   upscaler.

Both corrections use bounded per-channel mean and contrast matching. The
setting is persisted and the activity log records whether it is enabled for an
export.

## Model deployment

The app loads an unmodified, off-the-shelf model file:

- File: `models/interpolation/RIFE_fp32_timestep.onnx`
- Source: <https://huggingface.co/walterlow/RIFE_fp32_timestep>
- Pinned revision: `ee09066f9822f8b28b8477a1b4cc30f19d607590`
- SHA-256: `4DA60C1F20D42DBA4F21503140940AA48A488585EAD977532BF22E95A0319327`
- License: MIT

No conversion or preparation step is required. In a packaged build the model
belongs beside the executable under `models/interpolation/`. If it is absent,
the UI says `model missing`, the interpolation switch cannot be enabled, and
an interpolated task cannot be queued.

The binary model payload is intentionally ignored by Git, matching the policy
for detail models. Its provenance, pinned revision, checksum, and placement
instructions are committed in `models/interpolation/README.md`.

## Processing order

```text
decode and source restoration
    → square-pixel/CFR normalization
    → exact output-time scheduling
    → hard-cut check
    → RIFE at source resolution
    → optional RIFE color match
    → detail upscaler
    → optional detail-model color match
    → detail limit and strength
    → requested output scale
    → NVENC video and copied source audio
```

RIFE runs at source resolution so interpolation does not pay the much larger
cost of processing already-upscaled frames. Detail upscaling still runs once
for every final output frame, so increasing 24 fps to 60 or 120 fps increases
the total detail-model workload substantially.

## Frame scheduling

For each requested output timestamp, Kraken finds the two surrounding source
frames and sends their exact fractional phase to the timestep input. This
supports non-integer conversions such as 24→60 fps directly; it does not rely
on recursive midpoint generation.

The final source frame is held only for the short uncovered tail required to
preserve the original duration. The encoder receives the selected target rate,
and queue progress uses the interpolated frame count.

Preview positions that land exactly on a source frame show the midpoint of the
next source pair. This makes the enabled RIFE stage immediately visible while
the export scheduler continues to use exact output timestamps.

## Hard cuts

Adjacent frames are sampled for a normalized mean absolute difference. When a
pair crosses the fixed hard-cut threshold, Kraken selects the nearer endpoint
instead of asking RIFE to morph unrelated scenes. Color matching across that
cut is skipped as well.

## Interface and logging

- The section is labelled `Frame Interpolation — RIFE` when the model exists.
- The target selector offers 48, 60, and 120 fps.
- Output badges identify RIFE and the target rate.
- Queue rows show final FPS and final frame count.
- Output filenames include `_<fps>fps` when interpolation is enabled.
- Preview completion records the RIFE phase and `color matched` state.
- Export-start entries record output frame count, FPS, RIFE, and Color Match.

## Verification record

- The pinned model loaded and completed a real CUDA inference.
- A full-size 704×1248 pair from `SpaceVoyage_2.mp4` completed RIFE inference
  plus color correction in 0.31 seconds on the RTX 3090.
- A shortened test asset derived from `SpaceVoyage_2.mp4` completed the full
  decode → arbitrary-timestep 24→60 RIFE → Color Match → detail upscale →
  NVENC pipeline.
- The resulting file reported the expected 60 fps rate, frame count, and
  dimensions.
- A live release preview reported:
  `RIFE 60 fps · t=0.50 · color matched · fp16 · 2.8s on CUDA`.
- Standard tests, strict lint checks, and the optimized release build passed.

Relevant regression tests:

- `interpolate::tests::runs_real_model`
- `interpolate::tests::runs_full_size_clip_pair_with_color_match`
- `export::tests::exports_real_rife_sequence`
- `upscale::tests::color_match_corrects_channel_drift`

## Deliberate limits

- Slow-motion output is not exposed because retaining source FPS would change
  duration and requires a defined audio policy.
- Temporal detail locking is separate from frame interpolation and remains
  planned under Stability.
- IVTC and grain extraction/reapplication remain separate planned source
  features.
