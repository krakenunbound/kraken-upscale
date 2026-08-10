# Generative model packages

Kraken's Generative Refine mode loads ordinary stable-diffusion.cpp-compatible
weights without converting them. Each JSON file in this directory describes a
complete package. Paths may be absolute or relative to the JSON file.

FLUX.1 requires four components:

- diffusion transformer (`.safetensors` or `.gguf`)
- VAE
- CLIP-L text encoder
- T5-XXL text encoder

The included `flux1-dev.json` shows the expected layout. Model payloads are
local deployment files and are intentionally ignored by Git.

The beside-app runtime is the official Windows CUDA 12 build of
`stable-diffusion.cpp` release `master-813-bfbef5b` plus its official CUDA 12
runtime archive. The downloaded CUDA ZIP SHA-256 is
`E101FCD3AB323547EF8B4387B5EDC6E4A1A70837D394CA744CD847A30E3A9A71`.
Its MIT and ggml license files are kept inside `native-diffusion/`.

- Project: <https://github.com/leejet/stable-diffusion.cpp>
- Pinned release: <https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-813-bfbef5b>

On the development workstation, the four FLUX.1 Dev payloads are NTFS
hardlinks to the already-downloaded files under `F:\Kraken Art\models`. They
occupy no second copy of the data. A portable copy of Kraken still needs the
actual files beside this manifest.

No conversion is required: download a supported model's ordinary weights,
place all required components in a subfolder here, and add a manifest modeled
after `flux1-dev.json`. Incomplete packages stay out of the picker.
