# NVIDIA RTX VSR in Kraken

Kraken's **Images → RTX VSR** option calls NVIDIA RTX Video Super Resolution
directly. It is the native equivalent of the tested workflow:

```text
Load Image → RTXVideoSuperResolution (2×, ULTRA) → Save Image
```

It is not a diffusion model, checkpoint workflow, or Ultimate SD Upscaler. It
is intended for fast, faithful enhancement with compression cleanup and no
prompt-driven invention.

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

These files are NVIDIA proprietary software and are intentionally ignored by
Git. Anyone redistributing a packaged build is responsible for complying with
NVIDIA's runtime license.

## Controls

- Scale: x2, x3, or x4.
- Quality: Low, Medium, High, or Ultra.
- Ultra matches the supplied `RTX-VSR-PRO.json` workflow.
- Output: lossless PNG at the exact requested dimensions.

For outputs whose longest edge is above 15,360px, Kraken uses a protected VSR
intermediate and an antialiased final resize. Outputs above 16,384px are
rejected. A sampled RGB integrity check stops the known catastrophic 16K
single-channel failure from being saved.

## Verification

The native opt-in test is:

```text
cargo test rtx_vsr::tests::hardware_smoke_processes_rgba_on_the_gpu -- --ignored
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
