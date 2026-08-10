# Kraken Upscale — design decisions and why

Running record of the choices behind this app, including the ones that were
wrong and got reversed. Written so the reasoning survives being forgotten.

Last updated: 2026-08-10

---

## 1. What this is

A native Windows video upscaler. Target quality: match Topaz on clean-ish
sources, beat Aiarty everywhere.

**Target content — corrected 2026-08-07 after auditing the real library.**

The original brief was "old TV shows and movies, AI videos", which was read as
DVD-era standard definition. That was wrong. The actual material is:

| Source | Form |
|---|---|
| TV series rips (e.g. Star Trek Voyager) | 1440×1080 or 1920×1080, H.265 **10-bit**, progressive, 23.976 |
| Film rips | Blu-ray, 1080p, film grain intact |
| AI-generated video | 704×1248-class, clean, 8-bit |

**There are no DVDs in the collection.** ~99% of use is downloaded rips.

This reorders the work substantially:

- **10-bit precision is the dominant quality gap.** The pipeline decodes to
  `rgb24`, so a 10-bit source is quantised to 8 bits before the model sees it,
  and encoded back out at 8 bits. Banding appears exactly where this content
  lives — space, nebulae, dim interiors.
- **Broadcast restoration is insurance.** IVTC and deinterlace are built and
  tested (§8) but will rarely fire on progressive Blu-ray-era material.
- **Grain handling rises in importance**, because film-sourced Blu-rays carry
  genuine grain rather than the compression noise SD material carries.
- **Detail models must be chosen for already-upscaled sources.** Several of
  these rips are themselves upscales of SD masters, so they carry someone
  else's artifacts. Conservative models (Remacri, RealESRGAN) are safer than
  aggressive ones (UltraSharp, Siax) which sharpen those artifacts.

Hard requirements from the outset:

Hard requirements from the outset:

- A genuine self-contained Windows program, not a web app.
- Users download models from HuggingFace / OpenModelDB, drop them in a folder,
  press a button. No conversion step, no toolchain, no setup.

---

## 2. The single most important technical fact

**The gap between a good image upscaler and a good video upscaler is temporal,
not spatial.**

Real-ESRGAN and community models already match or beat Topaz *on a single still
frame*. Topaz wins because frame 100 and frame 101 agree with each other.

This matters because the appealing part of image upscaling — invented, plausible
micro-detail — is generated **independently per frame**. On video the model
re-invents pores, fabric weave and foliage 24 times a second, and the surface
appears to boil. The better the model, the worse the boiling: a weak upscaler
adds little and stays calm, a strong one adds gorgeous detail that shimmers.

**Correction, 2026-08-10:** a post-model temporal constraint is useful, but it
is not equivalent to a learned video super-resolution model. The primary path
should consume neighboring frames inside the neural detail stage so motion and
detail are learned together. The architecture now has three honest model
classes:

> multi-frame video model (preferred) → post-processing
>
> video-tuned frame model (fast) → optional motion-guided temporal lock → post-processing
>
> still-image model → optional motion-guided temporal lock → post-processing

NanoVSR-644k is the first genuine video model. It consumes five frames as one
NTCHW tensor, propagates features forward and backward, and returns five 4x
frames. Long clips remain streamable through bounded windows; hard cuts close
and pad the current window so recurrent features never cross between shots.

The existing temporal lock remains valuable for the large library of RRDBNet /
ESRGAN image models, but it is a compatibility bridge. It is bypassed for true
video models because stacking both mechanisms suppresses genuine motion.

Real-ESRGAN AnimeVideo/General v3, Real-CUGAN, SPAN, and Compact models are
useful video choices without being temporal architectures. They are installed
under `models/video-frame/`, labelled as fast frame models, and retain the
Stability lock. This preserves the useful performance/quality recommendation
without relabelling single-frame inference as multi-frame VSR.

### 2a. Still images are a separate product mode

The existing ESRGAN/RRDB models remain useful; they were simply in the wrong
workflow once Video mode was narrowed to video-appropriate choices. Kraken now
has a separate Images mode that opens still files, filters the model picker to
still-image models, previews one frame, and exports a lossless PNG. It shares
the native ONNX/CUDA tile engine and the spatial post-processing controls, but
does not expose Source cadence, temporal Stability, a timeline, or RIFE.

Transparency is not sent through a three-channel detail model. The source
alpha plane is resized independently and recombined after RGB inference. This
keeps logos, sprites, and artwork usable without pretending the detail model
understands transparency.

**Ultimate SD / checkpoint refinement is not the same stage.** A diffusion
checkpoint is not an upscale model that can be dropped into the existing NCHW
contract: it also needs a text encoder, VAE, denoiser, sampler/scheduler,
conditioning and seam-aware latent tiling. It is therefore implemented as the
clearly labelled Generative Refine submode behind the conventional image
upscaler, using a bundled `stable-diffusion.cpp` CUDA runtime and a complete
manifested model package. Kraken does not fake that workflow with an ESRGAN
model and a checkpoint filename.

**Kraken Art is the behavioral reference.** The grandfather project at
`F:\Kraken Art` implements the behavioral reference three-stage USDU workflow in
`python/pipelines/usdu.py`: target-aware overshoot and tile calculation,
ESRGAN pre-upscale plus model-specific FLUX/SDXL img2img refinement, then an
exact-resolution snap. Its Image Upscale UI also exposes ESRGAN/USDU modes,
checkpoint, VAE, steps, denoise, tile size, CFG, sampler, scheduler, prompt,
and a deterministic quality finish. That orchestration has now been ported.

The old runtime itself is not reusable here because Kraken Upscale remains a
self-contained native application. Its current implementation also uses
independent per-tile seeds and simple feather masks, while its calculated
seam-fix geometry is not consumed by `run_usdu`. Treat it as the functional
baseline. The native first pass already improves the paste with 64-pixel tile
context and raised-cosine overlap weights. Shared global latent noise and a
second seam-repair pass remain possible quality experiments.

---

## 3. Language and GUI

**Rust.** Working toolchain already present; `ort` gives mature ONNX Runtime
bindings; the C++ path needed vcpkg/ninja/MSVC setup before line one.

**Slint**, not egui / Iced / Tauri.

- Tauri was excluded on the "no web apps" requirement, even though it produces a
  real `.exe`. (Noted: the user's own MiniMax H3 Studio *is* Tauri, which is
  part of why this one shouldn't be.)
- egui is faster to build with but looks like a game-dev tool by default.
- Slint compiles declarative UI into the binary and renders via Skia/FemtoVG.

**Native OS window frame, not frameless.** Initially built frameless with a
custom title bar to match Aiarty. That was wrong: Slint does not add resize
borders to a frameless window, so **the window could not be resized at all**.
Reverted to a native frame with a branded toolbar underneath — which is also
what H3 Studio does. Native frame gets resize, snap and multi-monitor DPI free.

---

## 4. Inference backend: ONNX Runtime + CUDA, not NCNN/Vulkan

The original plan (from a Grok conversation) recommended NCNN + Vulkan for broad
GPU support with no CUDA lock-in.

**Rejected.** That is correct advice for shipping to strangers on AMD cards. On
a single RTX 3090 it buys portability that isn't needed today and costs exactly
the models that close the Topaz gap — recurrent and transformer VSR
architectures (deformable convs, recurrent state, DiT blocks) range from painful
to impossible in NCNN.

NCNN/Vulkan remains viable later as an alternate backend behind a trait, if this
ever ships to non-NVIDIA users.

### CUDA runtime discovery

ORT's CUDA provider loads `cublasLt64_13.dll` and cuDNN 9 by name. Rather than
requiring a system-wide CUDA toolkit install, `src/cuda.rs` prepends candidate
directories to the process PATH before any session is built:

1. `cuda/` beside the executable (the shipping layout)
2. `KRAKEN_CUDA_DIR` override
3. `CUDA_PATH/bin` (system toolkit)

Falls back to CPU with a loud log line rather than silently running 30x slower.

---

## 5. ffmpeg: LGPL build, which forces NVENC

Video I/O shells out to vendored ffmpeg binaries rather than linking libavcodec.
Linking the ffmpeg libraries on Windows is a real toolchain fight, and the
process boundary costs almost nothing in a streaming pipeline.

**We ship an LGPL build** (BtbN `n8.1-win64-lgpl-shared`), not the GPL Gyan
build that was already on the machine, so the app can be distributed.

**Consequence: no libx264 or libx265** — those are GPL. Encoding therefore uses
NVENC:

- `h264_nvenc` — verified working
- `hevc_nvenc` — verified working
- `av1_nvenc` — **fails on this hardware.** The 3090 is Ampere; AV1 *encode*
  needs Ada (RTX 40-series) or newer. The encoder exists in the build and errors
  at runtime with "No capable devices found".

Frames never touch disk: ffmpeg decodes to a raw rgb24 pipe, each frame goes
through the model, results are written straight to an encoder process. Audio is
copied from the original by adding it as a second input (`-map 1:a:0?`, optional
so clips without audio still work).

---

## 6. Model format — direct folder-drop loading

### Where it stands

Community RRDBNet models such as UltraSharp, Remacri, Siax, AnimeSharp, and
RealESRGAN can be dropped into `models/` unchanged. The app's native importer:

- reads both modern ZIP checkpoints and legacy stream checkpoints directly;
- `ort`'s **Model Editor API** (`editor::{Node, Graph, Model}`) builds an ONNX
  graph in memory and hands back a live inference session;
- infers RRDBNet topology and native scale from state-dictionary keys and tensor
  shapes, including the RealESRGAN 2× input rearrangement;
- also loads `.onnx` models directly, with dynamic input/output name discovery.

Two key layouts exist and both must be handled (verified against real files,
both 702 tensors, nf=64, nb=23):

| Style | Keys |
|---|---|
| Old ESRGAN — UltraSharp, Remacri, Siax, AnimeSharp, NMKD | `model.0.weight`, `model.1.sub.N.RDB{1,2,3}.conv{1..5}.0.weight`, scale-dependent upsampling/HR/output layers |
| New RealESRGAN | `conv_first`, `body.N.rdb{1,2,3}.conv{1..5}`, `conv_body`, `conv_up1`, `conv_up2`, `conv_hr`, `conv_last` |

RRDBNet covers the large majority of what people download for upscaling. SPAN,
Compact and RealPLKSR would come later.

Verified unchanged files: RealESRGAN 2×/4×, UltraSharp 4×, Remacri 4×, and
NMKD Faces 8×.

### Model discovery

`models/` is the source of truth, not any index file. Dropping in a model makes
it selectable with no rebuild. `models.json`, when present, only supplies
metadata that would otherwise be probed — chiefly the native scale factor. A
model with no metadata gets its scale **probed** by running a 32×32 input and
measuring the output.

### Licensing note

Real-ESRGAN is BSD-3. UltraSharp, Remacri and Siax are community models with
their own terms, some non-commercial. This matters if the app is ever
distributed with models bundled.

---

## 7. Quality controls, and why "denoise" and "steps" don't exist here

A recurring confusion worth recording. In ComfyUI, Ultimate SD Upscale exposes
**steps** and **denoise**. Those are **diffusion** concepts: denoise controls how
far the sampler may wander from the input *during generation*.

ESRGAN-family models are **feed-forward GANs** — one convolutional pass,
deterministic, no noise schedule. There is nothing to step and no denoise
parameter to expose. Their absence is architectural, not an oversight.

What exists instead, in the order it is applied:

### Detail limit (`clamp_deviation`) — the useful one

Caps how far any pixel may move from the source. Structural mistakes (freckles
redrawn as black strokes) are *large* excursions; genuine sharpening is a
*small* one. Clamping removes the former and leaves the latter intact.

### Strength (`blend_toward_source`) — the blunt one

Linear blend of finished model output toward a resampled source. It attenuates
**everything** equally, so pulling it down to kill an artifact also discards the
sharpening you wanted. At 0.5 you do not get freckles back — you get
50%-opacity black lines over a softer image. It does visibly reduce artifacts;
it just cannot repair a structural error.

### Scale — and a bug worth remembering

The model runs at its **native** factor (usually 4x) and the result is
**area-averaged down** to the requested scale last.

This ordering is deliberate: downsampling averages away invented
high-frequency detail, so *a 4x model asked for 2x is visibly cleaner on skin
than raw 4x output*. It is a real artifact-suppression lever, not just a resize.

For a while the Scale control **did nothing** — the model ran at 4x and only the
badge changed, so raw 4x output was being shown regardless of the setting.

### Scale should have two modes (planned)

Currently scale is multiplier-only (`x2 / x3 / x4`). It should offer:

1. **By multiplier** — `x2`, `x3`, `x4`. What exists now.
2. **By target format** — HD (1280×720), Full HD (1920×1080), 2K (2560×1440),
   UHD/4K (3840×2160), and possibly DCI 4K (4096×2160).

Target-format mode is cheap to implement because the pipeline **already
resamples last**: the model runs at its native factor and the result is
area-averaged to whatever size is asked for. Non-integer scale factors
therefore cost nothing extra — `720×480 → 3840×2160` is the same code path as
`x2`.

Useful reference points:

| Source | ×2 | ×3 | ×4 |
|---|---|---|---|
| 720p (1280×720) | 2560×1440 | **3840×2160** | 5120×2880 |
| DVD NTSC (720×480) | 1440×960 | 2160×1440 | 2880×1920 |
| 1080p (1920×1080) | **3840×2160** | 5760×3240 | 7680×4320 |

720p ×3 and 1080p ×2 land *exactly* on 4K UHD. ×4 from 720p overshoots 4K by
78% more pixels for no display benefit — which is an argument for offering
target-format mode as the default way to choose.

**The trap: aspect ratio and non-square pixels.**

DVD-era content is 4:3, so it does not map onto any 16:9 target without
pillarboxing or cropping. Worse, DVD stores **non-square pixels**: NTSC DVD is
720×480 stored, but displayed as 4:3 (or 16:9 for anamorphic discs) via a
display-aspect-ratio flag. The stored pixel grid is not the shape the picture is
meant to be.

So target-format mode must work from the **display** aspect ratio
(`ffprobe` `sample_aspect_ratio` / `display_aspect_ratio`), not the coded frame
size, and must decide explicitly between pillarbox, crop, or stretch. Getting
this wrong on Star Trek DVDs produces subtly squashed people that are hard to
notice until compared side by side.

### Model choice is the biggest lever of all

Picking the wrong architecture for the content is the most common cause of
visible artifacts, and a filename does not tell you that. Hence per-model
guidance text in the UI rather than a generic blurb.

The clearest case: **AnimeSharp on a photographic face draws dark ink strokes
where freckles are.** It is trained on line art. Scattered, organic squiggles
(as opposed to straight lines on a tile grid) are the signature.

Default model is chosen by a preference ranking, **not alphabetically** —
alphabetical order put AnimeSharp first, the worst possible default for
photographic footage. Remacri is the default.

---

## 8. Source preprocessing — why it is half the product

Two target contents need near-opposite treatment.

**AI-generated video** is the easiest case in the field: clean, no compression
damage, no grain, no interlacing, and diffusion output is inherently soft — so
detail synthesis helps immediately, and optical flow is reliable.

**Old TV and movies** carry three landmines, all of which make a detail model
*actively worse* than doing nothing:

1. **Interlacing.** Much pre-HD TV is 480i. Feed interlaced frames to an
   upscaler and it renders comb teeth into razor-sharp comb teeth.
2. **Telecine.** Scripted TV and film were shot at 24fps and padded to 30 for
   NTSC via 3:2 pulldown. Without inverting it, ~20% of frames are combed
   hybrids of two moments. Soaps, news and much British TV were shot true
   interlaced and need the opposite handling — so this needs *detection*, not a
   fixed setting.
3. **Compression artifacts.** DVD-era MPEG-2 blocking and mosquito noise. A
   detail model does not know a blocking artifact isn't a real edge, and will
   sharpen it into a crisp permanent rectangle.

Plus **grain**: it is high-frequency detail, so models either eat it (plastic
skin) or amplify it into crawling static. It needs separating out, holding
aside, and re-applying afterwards.

Hence two paths through one pipeline:

- *AI video* → detail model → temporal lock → encode
- *Old TV/film* → detect fields & telecine → IVTC or deinterlace → deblock →
  degrain (hold aside) → detail model → temporal lock → re-inject grain → encode

**Star Trek specifically:** DS9 and Voyager never got a film rescan and exist
only in 480i SD, which is exactly where this matters most. They were shot on
35mm and telecined for the video edit, but effects shots were composited on
video — so field cadence must be detected **per scene, not per file**.

---

## 9. UI bugs worth remembering

**Word-wrapped `Text` reports its unwrapped height to the layout.** A
three-line help string reserved one line, so every following section was laid
out on top of it. Cost three wrong fixes (`preferred-height`, `min-height`, and
a cyclic `viewport-height` binding) before an oversized-viewport test proved the
ScrollView was never involved. Fixed with an explicit `lines` property that
states the wrapped height up front.

**A trailing `TouchArea` swallows clicks aimed at its siblings.** Later siblings
paint on top in Slint. The section header's TouchArea was declared after its
row, so the gating checkboxes could not be clicked at all. Declare it *first*.

**Cyclic bindings are dropped silently.** Deriving a ScrollView's
`viewport-height` from something inside its own subtree makes the viewport
depend on its content; Slint drops the binding and falls back to the visible
height, with no error. Section open/closed state was hoisted to the window root
to break the cycle.

**Detail Model had a checkbox that could not be unchecked.** Beyond the
click-swallowing bug, gating it was meaningless — the detail model *is* the app.
Removed.

**Icons are vector paths, not font glyphs.** Segoe UI has no coverage for most
geometric/symbol codepoints and rendered them as tofu boxes. Shipping an icon
font for a handful of marks wasn't worth it.

**The before/after divider tracks the displayed image, not the pane.**
`image-fit: contain` letterboxes the frame, so a pane-relative divider drifts
off-centre on any clip whose aspect differs from the viewport.

---

## 10. Left dock — logs and system panels

Modelled on PixInsight's anchored side tabs: a 24px strip pinned to the left
margin is always present, and clicking a tab slides a 380px panel out from
behind it. Tab labels are rotated 90° (`Text` supports `rotation-angle`);
**panel content stays horizontal**.

Two panels:

- **LOGS** — timestamped activity, capped at 400 lines. Rendered **newest
  first** on purpose: it puts the live line at the top and removes any need for
  autoscroll logic, which would otherwise fight the user whenever they scrolled
  back to read something.
- **SYSTEM** — live GPU telemetry (usage, temp, clock, power, VRAM used/free,
  driver) plus runtime versions (ort, ONNX Runtime, active provider, CUDA,
  cuDNN, ffmpeg) and the model count and directory.

GPU telemetry comes from `nvidia-smi`, polled **on a background thread**, not a
UI timer — each call takes 50–100ms, which would be a visible stutter on the UI
thread. The poller exits when `invoke_from_event_loop` starts failing, so it
does not outlive the window.

`provider` shows what ORT *actually registered*, updated when a model loads,
rather than what was requested — CUDA can silently fall back to CPU.

### Tab labels: stacked letters, not rotated text

First attempt used `Text` with `transform-rotation: -90deg` (note:
`rotation-angle` is deprecated, and `slint-build` promotes that deprecation
warning to a **build error**).

The result was unusable — rotated glyphs resample out of the font atlas and come
out mushy and hard to read at tab sizes, and letter-spacing made it worse.
Replaced with the `VTab` component, which stacks upright capitals vertically.
Renders at native crispness, reads fine for short labels.

**Unverified:** the tabs themselves render correctly, but the panel actually
sliding open has not been confirmed. A synthetic click on the SYSTEM tab did
not visibly expand the dock. Suspect the animated `width` on a child of a
`HorizontalLayout` is being overridden by the layout — if so, the fix is to
drive `min-width`/`max-width` instead of `width`. **Check this first.**

## 11. Settings persistence

Stored at `%APPDATA%\KrakenUpscale\settings.json` — not beside the executable,
since a Program Files install is not user-writable.

The selected model is persisted **by name, not by index**. The models folder is
scanned at startup, so a file being added or removed would shift indices and
silently select a different model.

Saved on every `settings-changed` (the same signal that triggers a preview
re-render) and again on window close, which catches dock and view-mode state
that deliberately don't trigger a re-render.

## 11a. Performance — unresolved

Roughly **8 seconds per frame** at 4x on a 704×1248 source, RTX 3090, CUDA.
That is ~16 minutes for a 5-second clip, and impractical for a 45-minute
episode.

FP16 was implemented and measured at **10.8s vs 8.1s — no improvement**. Both
timings include CUDA session creation (~2–3s), which swamps the difference, so
**the comparison is currently meaningless**. Inference must be timed separately
before drawing any conclusion.

Note on FP16 export: post-hoc casting of an fp32 graph
(`onnxconverter-common`) produced graphs ORT refused to run. Tracing a genuinely
half model works, but the graph must be wrapped so **inputs and outputs stay
fp32** — otherwise the application has to feed f16 buffers.

### Why this is slower than Aiarty

Measured against Aiarty Video Enhancer on the same clip and it is significantly
slower. Causes, largest first:

1. **A 4x model is being run for a 2x output** (see §7). A native 2x model saves
   work, although the measured RealESRGAN_x2 gain on the representative clip
   was only about 17% (3.5s versus 4.2s), not the theoretical 4x.
2. **RRDBNet is heavy and old.** ESRGAN-family: 23 RRDB blocks × 64 channels,
   ~16.7M parameters, 2018-era. Aiarty's `moDetail-HQ v3` is proprietary but
   almost certainly a modern lightweight net — SPAN/Compact-style architectures
   reach comparable quality at **10–50x less compute**. No amount of optimising
   RRDBNet closes an architectural gap that size.
3. **No TensorRT** — plain CUDA EP leaves 2–4x unclaimed.
4. **Tile size was conservative.** This is now measured on the target 3090 and
   the 704×1248 `SpaceVoyage_2.mp4` UltraSharp frame: 512/640/896/1024px took
   4.57/4.36/2.56/2.68 seconds. Auto mode now uses 896px on 20+ GB cards, with
   a scale-aware cap that keeps native 8x model outputs at or below 4096px.
5. **No tile batching** — tiles run one at a time, so the GPU idles between
   launches.
6. **Post-processing was single-threaded.** Color matching, clamp, blend, and
   resampling now use the CPU thread pool. Tile assembly remains serial because
   overlapping feather regions share accumulators.

| Fix | Expected gain |
|---|---|
| Match model native scale to target (RealESRGAN_x2 for 2x) | ~1.2x measured |
| 896px tiles on the RTX 3090 | ~1.7x measured |
| Batch tiles | ~1.5–2x |
| TensorRT EP | ~2–4x |
| Parallelise the Rust loops | Landed; modest |
| **Modern lightweight architecture (SPAN / Compact)** | **10x+** |

The first four compound to plausibly 10–20x without changing models. But
genuinely matching Aiarty's speed *at quality* means a modern architecture, not
a 2018 one — which is another reason the native `.pth` importer matters: it lets
SPAN and Compact models be dropped in and tried.

---

## 12. Honest ceiling

On clean-ish 1080p→4K this can match Topaz. On badly degraded VHS-era footage,
Topaz Starlight is diffusion-based and genuinely ahead; beating it there means
hosting a diffusion model that consumes the whole GPU at seconds per frame.

SeedVR2 3B/7B is already on this machine but is a diffusion transformer in
fp8/GGUF — it belongs in ComfyUI as a "hero shot" tool for short clips, not in a
Rust `.exe`.

### The temporal lock smooths real change as well as invented change

**This is a limitation of the approach, not a bug, and it cannot be engineered
away.** Recorded here so it is not repeatedly rediscovered and "fixed".

The lock blends the warped previous output into the current frame. That is a
temporal filter, and a temporal filter has no way to tell *invented texture that
changed* from *real content that changed* — both are simply differences between
consecutive frames at similar spatial frequencies.

Measured on `SpaceVoyage_2` with Lock 0.75, using a strength-0 export through
the identical pipeline as the no-invented-detail floor:

| | flicker |
|---|---|
| floor (no invented detail) | 1.348 |
| unlocked | 1.613 |
| locked | **1.203** |

Locked output sits *below* the floor. It removed all of the model's added
flicker (0.265) and then a further 0.145 of change that was genuinely in the
source.

Two things follow:

1. **Confidence gating does not solve it.** The gate releases on low match
   quality, occlusion and displacement — none of which are present on
   near-static content, which is exactly where the over-smoothing is worst.
   A motion-based release (16 px, §see `temporal::MOTION_RELEASE`) helps the
   *smearing* case and does nothing for this one.
2. **Only Lock strength trades it off**, and the right value is a perceptual
   judgement. A residual metric cannot distinguish "held still correctly" from
   "smoothed away detail I wanted" — both reduce frame-to-frame difference, so
   optimising the metric drives the setting to 1.0, which is wrong.

The preview renders the preceding frame so this can be judged by eye
(§10 of CHANGELOG, A1). Choosing the shipped default is an open item.

---

## 13. Open work

### Verified working end to end

A full export was validated against `SpaceVoyage.mp4` (704×1248, 24fps, 120
frames):

- Output **1408×2496** — exactly 2x, correct
- **All 120 frames**, 5.000s, 24fps preserved
- **AAC audio copied intact** (158 packets, 5.0s) — the remux path works
- 19.5 minutes wall clock, ~9.8 s/frame

That validates decode pipe → model → tiling → post-process → encode pipe →
audio remux as a complete chain.

### Open work

| Item | State |
|---|---|
| Native `.pth` importer | **Complete** — direct legacy/ZIP RRDBNet loading, 2×/4×/8× verified |
| Dock panel slide-out | **Verified** — automated UI test confirms 30→410→30 animation |
| Temporal lock (flow-guided stabilization) | **Complete** — flow, confidence gate, export and preview; default Lock value still to choose (§12) |
| Source preprocessing front-end | Partial — profile-driven deinterlace/deblock active; IVTC/grain remain |
| Scale by target format (HD/2K/4K) | Designed (§7), not built |
| Performance pass | 896px RTX 3090 profile and parallel post-processing landed; batching/TensorRT remain |
| Export cancel button | **Complete** — stops the active batch and removes partial output |
| Auto tile size checkbox | **Complete** — persisted and VRAM-aware |
| Frame interpolation (RIFE) | **Complete** — native CUDA ONNX stage, arbitrary 48/60/120 fps, cut protection |
| Stability controls | **Complete** — real controls, previewable |
| No-cut-ghosting verification | **Blocked** — no test clip contains a hard edit |

### RIFE deployment

Frame interpolation uses the MIT-licensed `RIFE_fp32_timestep.onnx` graph from
`walterlow/RIFE_fp32_timestep`, pinned by revision and SHA-256 in
`models/interpolation/README.md`. It accepts two RGB frames and a scalar phase,
so non-integer conversions such as 24→60 fps are evaluated directly rather
than recursively. The app loads the unmodified download through its existing
CUDA provider from `models/interpolation/`; no preparation tool is involved.

RIFE runs before detail upscaling to keep the interpolation graph at source
resolution. Hard cuts select an endpoint instead of generating a morphed
transition. With Color Match enabled, the generated frame is matched first to
the corresponding source-frame blend and again after detail upscaling. The
full-size 704×1248 RIFE pass plus color correction measured 0.31 seconds on the
target RTX 3090; the extra detail-upscale frames remain the larger cost of a
higher output frame rate.

### Where things live

| Path | What |
|---|---|
| `src/main.rs` | Wiring, workers, preview + export orchestration |
| `src/upscale.rs` | Model registry, ORT session, tiling, blend/clamp/resample |
| `src/pth.rs` | Direct RRDBNet checkpoint reader and in-memory graph builder |
| `src/export.rs` | Streaming decode→model→encode pipeline |
| `src/image_mode.rs` | Still-image probe, preview decode, CUDA inference, alpha preservation, and PNG export |
| `src/media.rs` | ffprobe metadata, single-frame decode |
| `src/gpu.rs` | nvidia-smi telemetry, runtime versions |
| `src/cuda.rs` | CUDA DLL discovery |
| `src/dnd.rs` | Native window-event file drop integration |
| `src/settings.rs` | Persisted settings |
| `ui/app.slint` | Layout |
| `ui/widgets.slint` | Components |
| `ui/theme.slint` | Palette — restyling starts here |
| `vendor/ffmpeg/` | LGPL ffmpeg binaries |
| `models/` | `.pth` and `.onnx` models, auto-discovered |
