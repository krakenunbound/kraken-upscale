# NVIDIA RTX VSR in Kraken

Kraken's **Images → RTX VSR** option calls NVIDIA RTX Video Super Resolution
directly through the Video Effects C API. It is not a diffusion model or
checkpoint pipeline. It is intended for fast, faithful enhancement with
compression cleanup and no prompt-driven invention.

Kraken asks VideoSuperRes for the **exact destination**, matching ComfyUI
RTX VSR Pro. Portrait 720-wide RGBA8 frames are padded to a 64-pixel /
256-byte pitch first. Hybrid bicubic is only the verified 15,360px 16K cap.
If this C DLL still returns NVIDIA error -9, the session retries at 2×.

## Runtime layout

The optional runtime lives beside the executable:

```text
Kraken Upscale.exe
rtx-vsr/
  NVCVImage.dll
  NVVideoEffects.dll
  nvVFXVideoSuperRes.dll
  nvngx_vsr.dll
  nvngxruntime.dll
  npp*64_12.dll
  nvinfer_10.dll
  nvinfer_plugin_10.dll
  nvonnxparser_10.dll
```

The current local package was sourced from NVIDIA's `nvidia-vfx` 0.1.0.1
distribution. Kraken uses only the native DLL payload; it does not import or
launch an external scripting runtime. The runtime requires a supported NVIDIA
RTX GPU and current driver. NVIDIA documents Windows driver 570.65 or newer.

These files are NVIDIA proprietary software. Download them from NVIDIA's
Video Effects / `nvidia-vfx` distribution and keep them in `rtx-vsr/`
beside the executable. Packaged builds must follow NVIDIA's runtime license.

## Video

**Video → RTX VSR** uses a persistent `VideoSuperRes` session: one effect,
exact target size, temporal history across frames, reset at hard cuts. RIFE
still runs first when interpolation is on. The Kraken Stability lock is
bypassed — NVIDIA already holds temporal state.

## Controls

- Output goal: HD 720p, Full HD 1080p, 2K 1440p, 4K/UHD 2160p, x2, x3, or x4.
- Quality: Low, Medium, High, or Ultra.
- Ultra is NVIDIA's maximum-detail VSR model.
- Images: PNG, JPEG 95, or lossless WebP at the exact requested dimensions.
- Video: same encode options as the model path (NVENC, audio copy).

For outputs whose longest edge is above 15,360px, Kraken uses a protected VSR
intermediate and an antialiased final resize. Outputs above 16,384px are
rejected. A sampled RGB integrity check stops the known catastrophic 16K
single-channel failure from being saved.

## Verification

The native opt-in test is:

```text
cargo test rtx_vsr::tests::hardware_smoke_processes_rgb_on_the_gpu -- --ignored
```

On the target RTX 3090 it processed a 640×360 RGB frame through Ultra and
returned a non-empty 1280×720 result. The effect and GPU image buffers are
destroyed after the operation. A final release UI test processed the supplied
720×1280 source to 1440×2560 through Ultra in 1.0 seconds on CUDA.

## Sources

- NVIDIA Video Super Resolution documentation:
  https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html
- NVIDIA Video Effects C API:
  https://docs.nvidia.com/maxine/vfx/latest/API/Reference.html
- NVIDIA's MIT-licensed C API headers and examples:
  https://github.com/NVIDIA-Maxine/Maxine-VFX-SDK
