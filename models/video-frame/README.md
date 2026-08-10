# Fast video-compatible models

These graphs were selected for video and animation work, but each graph still
processes one frame at a time. Kraken labels them `VIDEO · fast frame model`
and leaves the Stability stage available to reduce frame-to-frame shimmer.

The separate `models/video/` folder contains genuine multi-frame graphs that
consume neighboring frames together. Kraken labels those
`VIDEO · multi-frame` and bypasses the separate Stability lock.

Installed choices:

- Real-ESRGAN General x4v3 — compact general/AI-video option.
- Real-ESRGAN AnimeVideo v3 — compact animation-video option.
- Real-CUGAN Pro Conservative / Denoise 3 — 2D cleanup at two strengths.
- AniSD General / Cel SPAN — standard-definition animation variants.
- OpenProteus Compact — faithful live-action HD/FHD detail.
- ModernSpanimation V1 — modern animation and generated 2D video.

The files are ordinary ONNX downloads. No conversion step or external runtime
is required by Kraken. Provenance and hashes are recorded in `SOURCES.md`.
