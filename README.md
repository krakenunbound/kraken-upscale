# Kraken Upscale

Native Windows AI upscaler for video and still images. Rust + Slint UI, ONNX
Runtime CUDA, optional NVIDIA RTX Video Super Resolution, RIFE interpolation,
and optional FLUX-based generative refine.

This repository is **source and metadata only**. Model weights, CUDA/cuDNN,
FFmpeg binaries, NVIDIA Video Effects, and `stable-diffusion.cpp` runtimes are
downloaded or staged beside the executable. They are not in Git.

## Requirements

- Windows 10/11, x64
- NVIDIA GPU with a current Game Ready / Studio driver
- [Rust](https://rustup.rs/) 1.92+ (MSVC toolchain)
- CUDA-capable ONNX Runtime (pulled by the `ort` crate; extra provider DLLs
  may need to sit next to the built exe)
- FFmpeg/ffprobe on `PATH` or in a `ffmpeg/` folder next to the exe

## Build

```bat
cargo build --release
```

The GUI is `target\release\kraken-upscale.exe`. A portable launch copy can be
named `Kraken Upscale.exe` next to `ffmpeg\`, `models\`, and optional
`cuda\`, `rtx-vsr\`, `native-diffusion\` folders.

## Models

Place `.onnx` or RRDBNet `.pth` files under `models/`:

| Folder | Role |
|---|---|
| `models/` | Still-image / generic ESRGAN-family |
| `models/video-frame/` | Fast single-frame video models |
| `models/video/` | True multi-frame models (e.g. NanoVSR) |
| `models/interpolation/` | RIFE |
| `models/generative/` | JSON package manifests for FLUX img2img |

Hashes and upstream links live in `models/video-frame/SOURCES.md` and the
READMEs next to each folder. **Check each model's license before you
redistribute it.** FLUX.1 Dev is non-commercial and must never be committed.

## Third-party runtimes

- FFmpeg: LGPL — see `vendor/ffmpeg/LICENSE.txt`
- ONNX Runtime / CUDA / cuDNN: NVIDIA and Microsoft terms
- RTX VSR: NVIDIA Video Effects SDK (proprietary; see `RTX-VSR.md`)
- `stable-diffusion.cpp`: MIT — see `models/generative/README.md`

Do not commit those binaries.

## License

Application source in this repository is MIT (see `LICENSE`). Bundled or
downloaded models and vendor SDKs keep their own licenses.
