# NanoVSR video model

`NanoVSR-644k-T5.onnx` is the first genuine multi-frame detail model in
Kraken. It consumes five neighboring low-resolution frames together and
returns five 4x frames. Kraken processes long clips in bounded windows and
spatial tiles, so neither the model nor the application holds a full clip in
memory.

- Project: NanoVSR — Towards Real-Time Video Super-Resolution on Edge Devices
- Authors: Filip Pawlicki, Marcel Kandula, Marcin Pucek, Kamil Dobies
- Official source: https://github.com/filippawlicki/nanovsr
- Official checkpoint release: https://github.com/filippawlicki/nanovsr/releases/tag/v1.0
- Checkpoint: `nanovsr_644k.pth`
- Checkpoint SHA-256: `515EEC20FF589986FC0E0DA18B363962B931008EBD848D0C8FB7E256CE200CF7`
- Deployment graph: fused official architecture, five-frame temporal window,
  dynamic spatial dimensions, ONNX opset 18
- Deployment graph SHA-256: `BE74A0422A5B8E0539B4C14EBCBA0F5E863BB61B52AE8A279E53CF9122078851`
- External tensor data SHA-256: `872BC27A92B6F985EC846BEACAE12AD584F79774B7A4FDBAC58FE0BE2BFC41E9`
- License: MIT (see the official project)

The graph is intentionally placed in `models/video/`. Regular image models
remain in `models/`, while interpolation graphs remain in
`models/interpolation/`; this prevents incompatible model types from being
mistaken for one another during discovery.
