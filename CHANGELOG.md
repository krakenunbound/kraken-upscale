# Changelog

## Unreleased — curated fast video model set

- Added eight compact video-compatible ONNX choices: Real-ESRGAN General x4v3,
  Real-ESRGAN AnimeVideo v3, Real-CUGAN Pro Conservative and Denoise 3, AniSD
  General and Cel SPAN, OpenProteus Compact, and ModernSpanimation V1.
- Added a separate `models/video-frame/` registry so fast frame-based video
  models are not mixed up with still-image models or genuine multi-frame VSR.
- The picker now labels all three classes explicitly: `VIDEO · multi-frame`,
  `VIDEO · fast frame model`, and `IMAGE`.
- Model-specific help states that the fast choices still process one frame at
  a time and therefore keep Kraken's Stability lock available.
- Recorded exact upstream sources and SHA-256 hashes beside the bundle.
- Ran every added graph through actual CUDA inference at its declared native
  scale. An incompatible ModernSpanimation FP16 graph was rejected; its working
  FP32 graph is installed.
- Rebuilt the root `Kraken Upscale.exe` (41,116,160 bytes; SHA-256
  `CD17138FBFD5857FBEACBECD1D4C19BD0E5D2140745F5D43B9BDAF681E49D68D`).

## Unreleased — genuine multi-frame video super-resolution

Kraken now has a distinct neural video-detail path instead of describing an
image model plus smoothing as equivalent to video super-resolution.

- Added NanoVSR-644k, a bidirectional recurrent model that consumes five
  neighboring frames and returns five 4x frames in one inference window.
- Added native NTCHW model discovery under `models/video/`; image detail,
  video detail, and RIFE interpolation graphs can no longer be confused.
- Long exports remain streaming through bounded five-frame windows and 320px
  spatial overlap-save tiles.
- Partial final windows repeat the last source frame and discard padded output,
  preserving duration and frame count.
- Hard cuts close the active window before the new shot so recurrent features
  cannot ghost across edits.
- Preview now renders a genuine temporal neighborhood for video models.
- The motion-guided Stability lock remains available for image models and is
  bypassed for learned video models to avoid double temporal smoothing.
- Verified the actual graph on CUDA and exported a seven-frame
  `SpaceVoyage_2.mp4` sample through two windows, including EOF padding.
- Visually checked the retained sample: restored face, hair, tattoo, fabric and
  hand detail were clean, with no visible spatial tile seams.
- Rebuilt the root `Kraken Upscale.exe` (41,101,824 bytes; SHA-256
  `37FDA4756E5E0C590CA4B031493FF312D196866B8592D20BF4646C8A26E2F030`).

## Unreleased — inverse telecine, and a corrected picture of the target content

### Inverse telecine

Reversing 3:2 pulldown recovers the original 24 fps frames exactly; deinter-
lacing the same footage destroys them. Measured against a known original:

| treatment | result |
|---|---|
| inverse telecine | **PSNR infinite** — bit-exact recovery |
| deinterlace | **38.9 dB** — permanent loss |

That loss lands *before* the detail model, which then amplifies it.
Deinterlacing telecined film also leaves the output at 29.97 fps where one
frame in five is a blend, so a fifth of the upscaling budget goes on frames
that never existed.

The Source profile already encoded the distinction, so **TV — shot on film**
now inverse-telecines and **TV — shot on video** deinterlaces. The rate change
30 → 24 flows through the encoder rate, output frame count and progress total,
keeping the exact rational (`24000/1001`, not 24) so audio does not drift
across an episode.

### Field detection reads pixels, not metadata

A telecined MPEG-2 VOB flags itself `field_order=tt` correctly — but remux it
to MKV and **the flag is gone while the combing remains**. Remuxes are what
people actually have, and a test caught this immediately.

The control is labelled "Detect field order & telecine", so it now runs `idet`
and looks. Cached, so the decode pass is paid once. Requires a clear majority
rather than a bare one, so occasional false positives on fast motion cannot
trigger field processing on progressive film.

### Test assets, built rather than sourced

Producing DVD-era test footage was treated as blocked on owning a DVD. It was
not: applying 3:2 pulldown to a known 24p source produces exactly the cadence a
disc carries, and keeping the original frames gives ground truth a real rip
cannot.

- `assets/progressive24.mkv` — 12 frames of lossless FFV1 (0.89 MB). Tests
  apply the telecine themselves so no encode sits in the comparison path.
- A full NTSC DVD simulation was validated separately: 720×480, SAR 8:9,
  DAR 4:3, top-field-first, MPEG-2 at disc bitrate. `idet` reports 59 TFF / 0
  progressive; IVTC returns 23.976.

### Target content corrected — this changes priorities

The design assumed DVD-era SD as a primary source. **It is not.** The actual
library is Blu-ray and web rips: 1080p or 1440×1080, H.265 **10-bit**,
progressive, already at film cadence. There are no DVDs.

Consequences:

- **The broadcast front-end is insurance, not the main path.** IVTC and
  deinterlace are correct and tested, and will almost never fire.
- **10-bit is now the largest quality gap.** Decode is `rgb24` — 8-bit — so a
  10-bit source is quantised before the model ever sees it. Banding shows up in
  gradients: space, nebulae, dark interiors, which is most of this material.
- **8-bit H.264 output is a second bottleneck**, for the same reason.
- **Grain handling matters more than expected**, since film-sourced Blu-rays
  carry real grain that the detail model will either eat or amplify.

## Unreleased — two "blocked" items were not actually blocked

### No-cut-ghosting verified on real footage

The cut-reset path had only ever been exercised on synthetic solid-colour
frames, because neither test clip contained a hard edit. Rather than wait for
one, `assets/hardcut-test.mp4` was built by splicing the two source clips,
giving a genuine edit at a known position.

`hard_cut_releases_the_lock_on_real_footage` asserts exactly one cut is found,
that confidence is zeroed there, and that locked output is **byte-identical**
to fresh detail — nothing from the previous shot bleeds across. Runs in the
default suite; no GPU required.

This closes the second half of the Phase 1 acceptance bar.

### FP16 settled — 4.8× faster, not slower

The earlier comparison concluded fp16 was *slower* (10.8 s vs 8.1 s). That was
entirely warm-up noise: CUDA session creation and cuDNN autotuning sat inside
the per-frame average. With warm-up tracked separately:

| | model steady |
|---|---|
| fp32 | 8026 ms/frame |
| **fp16** | **1668 ms/frame** |

fp32 runs first, so the ordering is pessimistic for the result being claimed.
The margin is well beyond the ~2× tensor cores alone would explain, which
suggests cuDNN also picks a poor fp32 algorithm at 896 px tiles. FP16 is not
optional on this hardware, and the default being on is correct.

## Unreleased — project audit (A1–A5)

Fresh audit of the tree after the B33–B42 closeout. Findings were mostly
product gaps rather than defects; the code itself is clippy-clean.

### A1 — the temporal lock is now visible in the preview

The lock was export-only, so Lock and Cut detect could only be judged by
running a full export and waiting minutes. For a feature that is on by default
and is the main quality differentiator, that made its controls unusable.

The preview now renders the preceding frame too and shows the locked result,
reporting the strength and the confidence actually held — e.g.
`lock 0.75 × held 0.80`. It runs only when Stability is on and the playhead is
off frame zero, and roughly doubles preview time, which the help discloses.

Post-processing is factored into one closure so both frames get identical
treatment; locking against a differently-processed previous frame would hold
the wrong pixels.

### A2 — confidence releases with displacement

Held detail belongs to a differently-lit, differently-occluded view of the
surface, so beyond some distance carrying it forward smears rather than
stabilises.

Threshold set from measurement: the estimator tracks cleanly to ~3 px (30–58%
residual reduction) and degrades at 9.8 px (17%), so **16 px** leaves good
tracking alone and releases where tracking is known to fail. Measured effect
matches the intent — near-static pairs barely move (0.84 → 0.83) while the
worst-tracked pair drops furthest (0.34 → 0.30).

**This does not fix the over-locking measured on near-static content**, and the
code says so. That is inherent: the lock is a temporal filter and cannot
distinguish invented texture that changed from real content that changed. Only
Lock strength trades it off — which is why A1 mattered more.

### A3 — halved the per-frame copy

The lock carried its source by cloning; it now reuses the previous buffer, so
one full-frame copy happens per frame instead of two.

### A5 — native-scale mismatch is no longer silent

A 4× model for a 2× target does ~4× the work and nothing said so. The Scale
control now warns, naming an installed native-scale alternative when one
exists.

### A4 — housekeeping

Removed the empty `tools/` directory left by the Python converter, and cleared
the last clippy warnings — including a doc block that had been orphaned onto
the thin `analyze()` wrapper instead of the `analyze_with()` that does the work.

## Unreleased — bug audit closeout (B33–B42)

All ten findings from the 2026-08-07 audit are closed. Nine were fixed; one did
not survive investigation.

### Fixed

- **B33** — runtime readiness derives from the CUDA payload rather than
  `nvidia-smi`. `GPU RUNTIME MISSING` was immediately overwritten by the probe,
  so a driver that could see a device reported `RUNTIME OK` even when ONNX
  Runtime had no provider it could load. `gpu::probe` now reports device
  presence only and the caller combines it with `cuda_dir`.
- **B35** — the Source help claimed AI video "skips deblock entirely" while
  Restore ran on every profile. Fixed the **copy**, not the gating: Restore is
  an explicit selection and silently discarding it because of an unrelated
  dropdown would be worse than honouring it. Deinterlace stays profile-gated
  because it is automatic.
- **B36** — before-pane frames decode on a worker thread. `media::frame_at`
  spawns ffmpeg and blocks, and running that on the Slint event loop froze the
  window on every scrub step. Generation-guarded on both the worker and the UI
  thread, and coalesced to the newest position.
- **B37** — the preview resumes when an export releases the GPU. Requests made
  during an export only bumped the generation, leaving the after-pane stale.
- **B38** — interpolated frame count scales the source count by the rate ratio
  when duration is unknown, instead of falling back to the source count and
  letting `-shortest` truncate the copied audio.
- **B39** — pipe hand-off failures tear down both children. `Child` does not
  kill on drop, so `?` there could orphan ffmpeg processes and their drain
  threads.
- **B40** — tile output is validated against the model's native scale, with the
  implied scale in the error. A mis-tagged `models.json` entry or a
  non-ESRGAN graph previously read past the buffer or corrupted the tile.
- **B41** — `.pth` import mirrors the `.onnx` provider policy: CUDA with
  `error_on_failure`, explicit CPU fallback, and the label taken from what was
  registered rather than `CUDA::is_available()`, which reports compile-time
  support and not what the session actually got.
- **B42** — export progress stops quoting a total once frames written exceed
  the probe estimate, rather than showing `134/120` with the bar at 100%.

### Closed as not a bug

- **B34** — filed as "the lock estimates flow from RIFE intermediates rather
  than original decoded pairs". That describes the behaviour accurately but the
  conclusion is wrong: output N comes from rendered frame N whether decoded or
  synthesized, so motion between consecutive outputs *is* motion between
  consecutive rendered frames. Using the original pair would be the real defect,
  since interpolated outputs sit at intermediate times.

  Measured to be sure — `lock_with_interpolation_does_not_regress` exports the
  same clip at 60 fps with Stability off and on: flicker **1.641 → 1.401
  (−14.6%)**. The combination improves output rather than smearing it. The
  reasoning is recorded at the lock site so it is not "fixed" into the wrong
  behaviour later.

## Unreleased — packaging

- Embedded the application icon in the executable, generated at 7 sizes
  (16–256 px) from the kraken artwork and compiled in via `winresource`.
  Explorer, the taskbar and Alt-Tab now show it. This is separate from the
  Slint `icon:` property, which only sets the in-app window icon at runtime.
  A missing resource compiler warns rather than failing the build.
- **Fixed a portability gap in the staged release.** `media::bin_dir` looks for
  `ffmpeg/` beside the executable and otherwise falls back to a compile-time
  absolute source path, so the staged copy only ran from this exact directory.
  The vendored binaries are now copied to `ffmpeg/` next to the exe, making the
  folder relocatable. `cuda/`, `models/` and the RIFE model already resolved
  beside-exe correctly.

### Staged release layout

```text
Kraken Upscale.exe
ffmpeg/                          ffmpeg + ffprobe and their DLLs
cuda/                            CUDA 13 + cuDNN 9 redistributables
models/                          upscale models (.onnx / .pth)
models/interpolation/            RIFE timestep model
onnxruntime_providers_cuda.dll
onnxruntime_providers_shared.dll
```

## Unreleased — Phase 1 step 3: lock wired into export

### Temporal lock is live

- The lock runs in the export pipeline and **Stability is a real control**, no
  longer a disabled placeholder.
- It runs **last, on the finished frame**, so what gets held is exactly what was
  written. Locking earlier would carry forward detail that later stages alter.
- Only the previous output and the frame it came from are retained, so export
  stays streaming regardless of clip length.
- Cut sensitivity is plumbed through instead of being hardcoded at 0.28, mapped
  inverted so a higher setting cuts more readily.
- The stage summary reports lock cost and how much was actually held, so a run
  where confidence collapsed everywhere no longer looks identical to one that
  worked.
- Preview still shows a single frame and is labelled as such — there is nothing
  for a lock to hold still in one frame.

### Measured — acceptance bar met, with a caveat

On `SpaceVoyage_2` (near-static, the content the bar targets), using a
strength-0 export through the identical pipeline as the no-invented-detail
floor:

| | flicker | model added |
|---|---|---|
| floor (no invented detail) | 1.348 | — |
| unlocked | 1.613 | 0.265 |
| locked | 1.203 | **−0.145** |

- **Model-added flicker removed: +154.6%** against a ≥50% bar.
- Wall cost **+2.2%** — the lock stage is 20 ms/frame, about 1% of total.
- On the moving clip the gate correctly holds back: confidence 0.71 vs 0.85,
  total flicker reduction 11.5% vs 25.4%.

**The overshoot is the real finding.** Locked output sits 0.145 *below* the
floor, so the lock is suppressing genuine source change as well as invented
detail. Benign on near-static content; the same mechanism smears motion. The
metric is deliberately left unclamped and warns about it — clamping would have
reported a flattering "100% removed" and hidden the problem.

## Unreleased — Phase 1 step 2: confidence gate

### Trust, then blend

- Added `temporal::analyze`, which estimates motion and decides per grid cell
  how far it can be trusted, and `temporal::lock`, which blends the warped
  previous output into current detail in proportion to that trust. Neither is
  wired into the pipeline yet.
- Two independent checks, chosen because they fail on different things:
  - **match residual** catches brightness change, where a fade leaves no good
    match at any offset;
  - **forward–backward consistency** catches occlusion and motion boundaries,
    which residual can miss when a wrong match happens to score well.
- They are **multiplied, not averaged**. Either failing alone makes the warp
  untrustworthy, and averaging would let a good residual mask an occlusion.
- Scene cuts disable the lock outright rather than softening it, so no invented
  detail is carried across an edit.

### Validated against the previously measured failure modes

The gate found both without being told about them:

| pair | residual reduction | confidence |
|---|---|---|
| 167→168 (9.8 px motion) | +17% — worst | **0.34** — lowest |
| 0→1 (brightness change) | +2% — worst | **0.45** |
| good pairs | +30…+53% | 0.79–0.84 |

- Cost rises from 5–8 ms to **13–15 ms** per pair, since flow now runs in both
  directions. Still under 1% of the 1.65 s/frame model budget.

## Unreleased — Phase 1 step 1: optical flow

### Temporal lock groundwork

- Added `src/temporal.rs`: motion estimation, warping, and the mean-absolute-
  difference metric the lock will be measured by. Not yet wired into the
  pipeline.
- Resolved ENHANCEMENTS decision checkpoint 1 in favour of a **built-in
  estimator** rather than a downloaded RAFT ONNX. Reasoning is in the module
  header: the lock is not optional the way interpolation is, so requiring a
  second download would mean most users never get it. The confidence gate, not
  flow accuracy, is what keeps poor estimates safe.
- Coarse-to-fine block matching on a luma pyramid, with a median filter over
  the vector field and a small zero-bias so flat regions resolve to no motion
  rather than noise.

### Measured

- Flow costs **5–8 ms per frame pair** — 0.4% of the 1.65 s/frame model budget.
- Residual falls on every pair measured and never rises.
- On `SpaceVoyage_4` (240 frames, real motion), sampling 10 pairs spread across
  the clip: **mean residual reduction +35.3%**, best +58%.
- `SpaceVoyage_2` is nearly static (0.02–0.76 px mean motion) and exercises only
  the flicker case; the measurement test now samples across the whole clip
  rather than taking six consecutive frames from the opening shot.

### Identified for the confidence gate

Two failure modes, both to be handled by declining to lock rather than by
improving the estimator:

- **Brightness change** — SAD is not intensity-invariant, so a fade or lighting
  shift leaves no good match. The clip's first frame pair reduced residual by
  only 2% despite 3.6 px of apparent motion.
- **Large or discontinuous motion** — at 9.8 px mean motion the reduction fell
  to 17%. Block matching is weakest at motion boundaries, and the median filter
  that removes outliers also smooths across genuine edges.

### Fixed

- **`frame_at` returned the wrong frame.** The seek time was formatted to
  millisecond precision, but a 24 fps frame spans 41.667 ms — so frame 2 begins
  at 0.083333 s while the request rounded to `0.083`, landing back inside frame
  1. Timeline scrubbing could show a stale frame. Found because two of six
  measured frame pairs reported exactly 0.00 residual while the same frames
  extracted sequentially were all distinct.

## Unreleased — Phase 0 export instrumentation

### Stage timing

- Added `src/stages.rs`: per-stage accumulator for the export pipeline
  (decode, interpolate, model, post, encode).
- Changed `export::run` to return the timing breakdown; the summary is written
  to the activity log when an export completes.
- Separated the first model call from the steady-state average. It carries
  one-off CUDA session creation and cuDNN autotuning that never recur, and
  folding it into the mean makes short runs look far slower than reality.
- Added `baseline_stage_timings`, an ignored test that measures an unlocked
  reference export and prints the breakdown, giving the temporal lock something
  concrete to be judged against.

### Measurement corrections

- **The previously recorded 2.56 s "model pass @ 896px tiles" is the one-off
  warm-up cost, not the per-frame model time.** Measured steady state on the
  same clip and tile size is **1.65 s/frame** — 36% faster than documented.
- The model stage is **98% of measured time**. Decode (5 ms), post (32 ms) and
  encode write (1 ms) per frame are negligible, so optimisation effort belongs
  entirely in inference.
- A first run immediately after a build measured 17.1 s/frame versus 1.77 s on
  the next identical run. Cold-start variance is roughly 10×, so any benchmark
  must discard its first run.

### Test environment

- ONNX Runtime provider DLLs are emitted to `target/release/` but GPU tests run
  from `target/release/deps/`. They must be copied across, and `KRAKEN_CUDA_DIR`
  must point at the repository `cuda/` folder, or tests silently fall back to
  CPU and fail an unrelated assertion.

## Unreleased — 2026-08-06 reliability and native-model pass

### Model loading

- Added direct `.pth` checkpoint loading in the Windows app.
- Added readers for both ZIP checkpoints and legacy PyTorch stream checkpoints.
- Added in-memory RRDBNet graph construction for old ESRGAN and new RealESRGAN
  state-dictionary layouts.
- Added native-scale inference for 2×, 4×, and 8× models.
- Added RealESRGAN 2× space-to-depth input handling.
- Added dynamic model input/output name discovery.
- Made the beside-app `models/` folder authoritative without requiring
  `models.json`; optional metadata remains supported.
- Removed the obsolete external model-preparation utility.

### Export reliability

- Added user cancellation for the active export batch.
- Guaranteed decoder and encoder kill/wait cleanup on cancel and all error paths.
- Removed partial output files after failed or cancelled exports.
- Continuously drained decoder and encoder diagnostics to prevent pipe stalls.
- Distinguished clean frame-boundary EOF from a truncated raw frame.
- Rounded output dimensions to valid even values for 4:2:0 encoding.
- Added no-overwrite encoder behavior and unique queued output paths.
- Prevented finished, running, or already-scheduled jobs from being queued twice.
- Changed queue scheduling to select the next unfinished task; the final primary
  button label is `Export`.
- Added `-shortest` to keep remuxed audio/video duration bounded.

### Media correctness

- Restored functional Source profiles: broadcast profiles can deinterlace
  detected fields, and the Restore selector applies light/strong deblocking or
  heavy-compression cleanup before inference.
- Kept every Restore option within the filters shipped by the bundled video
  engine and added execution coverage for all four selector values.
- Added real source-color matching with bounded per-channel mean/contrast
  correction in preview and export.
- Added a genuine RIFE timestep stage using a drop-in MIT-licensed ONNX model.
  It synthesizes motion-aware frames before detail upscaling at 48, 60, or
  120 fps, skips interpolation across detected hard cuts, and color-matches
  generated frames to their source-frame blend when Color Match is enabled.
- Added [RIFE-IMPLEMENTATION.md](RIFE-IMPLEMENTATION.md) with the permanent
  model provenance, pipeline order, scheduling, logging, performance,
  verification, and limitations record.
- Probed both average and nominal frame rates and retained the selected rational
  rate throughout decode and encode.
- Added explicit conversion to constant frame rate for the raw-frame pipeline.
- Probed sample/display aspect and normalized anamorphic sources to even,
  square-pixel display dimensions before inference.
- Applied the same geometry policy to preview frames and exports.

### Interface and lifecycle

- Added Clear Video and Clear Queue actions; clearing a running queue also
  requests cancellation, while clearing a video leaves queued snapshots intact.
- Restored the primary button label from `Export Next` to `Export`.
- Deferred new-video previews during an active export so the next video can be
  configured without competing for GPU time.
- Made missing GPU runtime a visible blocking state instead of silently starting
  an impractically slow CPU preview.
- Added preview start, model-load, per-tile progress/ETA, completion, and failure
  entries to the activity log.
- Mirrored activity entries to a persistent `activity.log` beside the saved
  settings so a failed or stalled session remains diagnosable after restart.
- Normalized unavailable sample-aspect metadata without showing a false warning.
- Replaced the conflicting legacy file-drop hook with the window framework's
  native Explorer drop events and added whole-stage hover feedback.
- Corrected compounding preview pan movement in split and side-by-side modes.
- Closing the main window now exits the event loop after saving settings and
  requesting active export cancellation.
- Replaced the nonfunctional hardware selector with an honest automatic-backend
  status.
- Replaced the inaccurate half-precision performance promise with neutral copy.
- Made automatic tile sizing real, persisted, and VRAM-aware.
- Kept Stability and grain preservation visibly planned and disabled until
  their pipelines exist.
- Activated the Frame Interpolation controls only when the RIFE model is
  present, and included target FPS/frame count in queued task details and logs.
- Made the fixed export-format footer visibly non-interactive.
- Changed unknown model scale display from `0×` to `auto`.

### Root launch build

- Built the optimized Windows executable and staged an easy-launch copy at
  `F:\Video Upscale\Kraken Upscale.exe`.
- Staged the CUDA provider DLLs and redistributable CUDA runtime beside the root
  executable; the existing root `models/` and `vendor/ffmpeg/` folders complete
  the local launch layout.
- Launched that exact root copy with `SpaceVoyage_2.mp4` and verified CUDA model
  loading, color-matched preview completion, model discovery, and a responsive
  window.
- Root executable SHA-256:
  `5ACD0B75C04CAC587A4594F5810B954B093B96799E6E25B939B05F1535D1B306`.
- Generated executable/runtime payloads remain outside Git; source, build
  instructions, model provenance, and change records stay committed.

### Verification

- Benchmarked automatic tile sizes on the target RTX 3090 using
  `SpaceVoyage_2.mp4`: 896px reduced the model pass from 4.57s to 2.56s versus
  512px.
- Raised the 24 GB automatic tile profile to the measured 896px optimum and
  parallelized color, blend, clamp, and resampling work across CPU cores.
- Made tiles scale-aware so native 8x models stay within a safe 4096px output
  pitch while 2x/4x models still benefit from larger tiles.
- Verified a real arbitrary-timestep 24→60 fps export through RIFE, color
  matching, detail upscaling, and NVENC, including output FPS and frame count.
- Measured full-size RIFE inference plus color correction at 0.31 seconds for a
  704×1248 frame pair on the target RTX 3090.
- Added tests for odd-dimension output sizing, exact EOF, truncated frame input,
  anamorphic media normalization, frame decoding, and dock width animation.
- Verified direct import and inference with unchanged RealESRGAN 2×/4×,
  UltraSharp 4×, Remacri 4×, and NMKD Faces 8× checkpoints.
- Confirmed the complete test suite and release build pass.
