# Generative model packages

Kraken's Generative Refine mode loads ordinary stable-diffusion.cpp-compatible
weights without converting them. Each JSON file in this directory describes a
complete package. Paths may be absolute or relative to the JSON file.

FLUX.1 requires four components:

- diffusion transformer (`.safetensors` or `.gguf`)
- VAE
- CLIP-L text encoder
- T5-XXL text encoder

The included `flux1-dev.json` shows the expected layout. Download the weight
files yourself and place them next to that manifest.

The beside-app runtime is the official Windows CUDA 12 build of
`stable-diffusion.cpp` release `master-813-bfbef5b` plus its official CUDA 12
runtime archive. The downloaded CUDA ZIP SHA-256 is
`E101FCD3AB323547EF8B4387B5EDC6E4A1A70837D394CA744CD847A30E3A9A71`.
Its MIT and ggml license files are kept inside `native-diffusion/`.

- Project: <https://github.com/leejet/stable-diffusion.cpp>
- Pinned release: <https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-813-bfbef5b>

You can NTFS-hardlink the VAE, CLIP-L, and T5 files from another local model
folder so they occupy no second copy. The Q6 diffusion transformer is a
separate download. A portable copy still needs all four actual files beside
this manifest.

### RTX 3090 package

Do not use the 23.8 GB FP32 diffusion transformer on a 24 GB card. Although the
native runtime can stream it through system memory, measured inference was
about 161 seconds per diffusion step and left the UI apparently motionless.

The active manifest instead uses city96's direct FLUX.1 Dev **Q6_K GGUF**. It
is an ordinary downloadable model file, not a Kraken-specific conversion:

- Source: <https://huggingface.co/city96/FLUX.1-dev-gguf>
- File: `flux1-dev-Q6_K.gguf`
- Size: 9,857,000,736 bytes
- SHA-256: `9566D56031D7F8DE184BB5A0393073956EC4E28B32DB3F860BD2B87EDCA04D13`
- License: FLUX.1 Dev non-commercial license, inherited from the base model

Q6 leaves room for the T5 encoder, CLIP-L, VAE, and working tensors on the
3090 while retaining substantially more model precision than Q4/Q5 variants.

No conversion is required: download a supported model's ordinary weights,
place all required components in a subfolder here, and add a manifest modeled
after `flux1-dev.json`. Incomplete packages stay out of the picker.
