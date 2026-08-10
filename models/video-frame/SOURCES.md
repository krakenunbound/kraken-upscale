# Model provenance

All hashes are SHA-256 of the exact files installed beside Kraken.

| Installed file | Source | SHA-256 |
|---|---|---|
| `realesr-animevideov3.onnx` | [vs-mlrt RealESRGANv3 model release](https://github.com/AmusementClub/vs-mlrt/releases/tag/model-20211209), exported from the [official Real-ESRGAN AnimeVideo v3 weights](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md) | `58425BDF2A28DF7109AEEFDC94D1B7366EAC85414F6CD6F2FBA41259F9D3307F` |
| `realesr-general-x4v3.onnx` | [CoderViking ONNX export](https://huggingface.co/CoderViking/realesr-general-x4v3-onnx), from the [official Real-ESRGAN release](https://github.com/xinntao/Real-ESRGAN/releases) | `1940A93EE08283A0A7286183186357B1688FE9FA8EDE74604B424586AADDF112` |
| `real-cugan-pro-conservative-2x.onnx` | [vs-mlrt CUGAN Pro model release](https://github.com/AmusementClub/vs-mlrt/releases/tag/model-20211209), implementing [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | `391A30605CA4101B54FF968728512B344EF277D8DBD1F686B12F0BF6C0C2D9E8` |
| `real-cugan-pro-denoise3x-2x.onnx` | same CUGAN Pro release | `39D45F6BD3969F69AF7348184FFACC4DFAC4609A6C23DBA4A3FC07548330BF76` |
| `2x-AniSD-General-SPAN.onnx` | [Sirosky AniSD release](https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD) (`2x_AniSD_G6i1_SPAN_215K_fp32.onnx`) | `EC446ACF6FDE4A23A27625EF1102D72F41EBA2C2F6A2E84A45848B986F08FEDF` |
| `2x-AniSD-Cel-SPAN.onnx` | same AniSD release (`2x_AniSD_AC_G6i2b_SPAN_190K_fp32.onnx`) | `C0D52AE736A41803AF7389911333F6A70F4B8F4BEE795C01713A3B8BD81B284D` |
| `2x-OpenProteus.onnx` | [Sirosky OpenProteus release](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus) | `133688B581809213D6A792C69AAE5C6CE7D10AEF859E593599D64D5974D25623` |
| `2x-ModernSpanimationV1.onnx` | [ModernSpanimation V1 release](https://github.com/TNTwise/Models/releases/tag/2x_ModernSpanimationV1) | `6CFE12C4C4FA68C60DCD0A33135A052E5A7FF451EA8C2921482661E2858F117C` |

Upstream project/release pages remain the authority for each model's terms and
intended use.
