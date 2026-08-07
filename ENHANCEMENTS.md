# Kraken Upscale — quality & performance enhancements

**Purpose:** ways to improve upscale quality and performance toward pro-level (Topaz-class quality, Aiarty-competitive speed).  
**Status:** findings only — no code in the originating session.  
**Last updated:** 2026-08-06  
**Ordering:** temporal lock **before** performance work.

Related: [DECISIONS.md](DECISIONS.md), [NEXT-SESSION.md](NEXT-SESSION.md), [RIFE-IMPLEMENTATION.md](RIFE-IMPLEMENTATION.md), [bugslist.md](bugslist.md), [CHANGELOG.md](CHANGELOG.md).

---

## 1. Executive summary

Spatial detail is already in the right league (Remacri, UltraSharp, RealESRGAN, folder-drop RRDBNet). The product is **not** blocked by “better still upscaling.”

To get as good as the pros:

1. **Make invented detail hold still across frames** (temporal lock) — **do this first**.
2. **Finish broadcast source restoration** (IVTC, grain separation/re-injection).
3. **Then** pay down the speed tax (tile batching, TensorRT, native-scale matching, SPAN/Compact).

> From DECISIONS §2: *The gap between a good image upscaler and a good video upscaler is temporal, not spatial.*

Strong per-frame models **increase** shimmer if each frame reinvents texture independently. Temporal constraint must be model-agnostic and written once in the pipeline.

---

## 2. Current baseline

### Working pipeline

```text
decode + restore/deinterlace
  → square-pixel / CFR normalize
  → optional RIFE (48/60/120) at source resolution
  → RRDBNet / RealESRGAN detail upscale (tiled, feathered)
  → color match → detail limit → strength blend → resample-last
  → NVENC + audio copy
```

Reliability pass is complete (see `bugslist.md`). End-to-end export validated on `SpaceVoyage.mp4`.

### Measured numbers (RTX 3090)

| Metric | Value | Source |
|---|---|---|
| Full export | ~9.8 s/frame; 19.5 min for 120 frames @ 2× UltraSharp | DECISIONS §13 |
| ~~Model pass @ 896px tiles~~ | ~~2.56 s~~ — **superseded, see below** | CHANGELOG / NEXT-SESSION |
| RIFE full-size pair + color | 0.31 s | RIFE-IMPLEMENTATION |
| RealESRGAN_x2 vs 4×→2× | ~17% faster only (3.5 vs 4.2 s) | DECISIONS §11a |

#### Phase 0 measured baseline (instrumented)

704×1248 source, 4x-UltraSharp at 896px tiles, 2× target, fp16, RTX 3090,
12 frames, no interpolation:

| Stage | Per frame | Share of measured |
|---|---|---|
| decode | 5 ms | 0% |
| **model** | **1649 ms steady** | **98%** |
| post (color/clamp/blend/resample) | 32 ms | 2% |
| encode write | 1 ms | 0% |
| wall clock | 1.77 s | — |
| model warm-up (one-off) | 2558 ms | excluded |

Three things this changes:

1. **The 2.56 s figure recorded earlier was the warm-up, not the model pass.**
   Steady state is **1.65 s/frame**, 36% faster than believed. Any prior
   reasoning anchored on 2.56 s was pessimistic.
2. **Inference is 98% of measured time.** Decode, post-processing and encode
   writes are collectively under 40 ms/frame. There is no meaningful headroom
   left in CPU post-processing — §4.4 items 3, 4 and the remaining speed work
   must target the model stage or nothing.
3. **Cold-start variance is ~10×.** The first run after a build measured
   17.1 s/frame against 1.77 s on an identical rerun. Benchmarks must discard
   their first run or they are noise.

Reproduce with:

```text
set KRAKEN_CUDA_DIR=F:\Video Upscale\cuda
set KRAKEN_TEST_CLIP=F:\MiniMaxH3\outputs\2026-08-06\SpaceVoyage_2.mp4
cargo test --release baseline_stage_timings -- --ignored --nocapture
```

### What already works well

- Open folder-drop models (`.pth` RRDBNet + `.onnx`)
- Overlap-feathered tiling; VRAM-aware auto tile (896px on 24 GB; scale-safe for 8×)
- Detail limit + strength (anti-hallucination)
- Resample-last (native model scale → target)
- RIFE timestep interpolation with hard-cut protection
- Source profiles: deinterlace (bwdif), deblock, bilateral restore
- Color match (bounded mean/contrast)
- CUDA ORT path, NVENC encode, audio remux

### Honest ceiling (documented)

- Clean 1080p→4K / AI video: **can match Topaz** once temporal lock lands.
- Badly degraded VHS-era / diffusion restoration: Topaz Starlight remains ahead until a separate diffusion “hero” path (SeedVR2 noted as ComfyUI, not in the native `.exe`).
- Speed: RRDBNet (~16.7M params, 2018) will not match lightweight proprietary nets without architecture change **and/or** TensorRT + batching.

---

## 3. Gap analysis vs pro tools

| Capability | Topaz / peers | Kraken today | Gap |
|---|---|---|---|
| Per-frame detail | Specialized models | Strong open ESRGAN-family | Small on stills |
| Temporal stability | Flow / recurrent / video models | **None** (Stability UI disabled) | **Largest quality gap** |
| Source restoration | IVTC, deinterlace, denoise, grain | Partial: bwdif + deblock/bilateral | Medium (broadcast) |
| Grain handling | Separate / re-inject | Planned only | Medium |
| Speed | Light nets + TRT / custom CUDA | CUDA EP, serial tiles, heavy RRDB | **Largest speed gap** |
| Model ecosystem | Closed curated | Open drop-in | Advantage if broadened |
| Frame interpolation | Built-in | RIFE done | Small |
| Target resolution presets | Yes | Multiplier only | Small |
| Face refinement | Optional stages | Faces model present, not staged | Medium |

---

## 4. Findings by area

### 4.1 Quality — temporal lock (P0)

**Problem:** Detail models invent plausible micro-detail independently per frame. On video, pores/fabric/foliage “boil.” Better models can shimmer worse.

**Designed solution (DECISIONS §2, not built):**

```text
for each output frame t:
  detail_t  = Upscale(source_t)
  flow      = OpticalFlow(source_{t-1}, source_t)
  warped    = Warp(detail_{t-1}, flow)
  conf      = FlowConfidence(flow, occlusion, cut)
  output_t  = conf * warped + (1 - conf) * detail_t
  # cuts / low conf → pure detail_t
```

| Piece | Approach |
|---|---|
| Flow | Drop-in ONNX (RAFT-small or lighter) on same ORT CUDA stack as RIFE |
| Warp | Correct bilinear warp of previous **output**; optimize later if hot |
| Confidence | Forward–backward consistency + magnitude + scene cut |
| Cuts | Reuse/extend `interpolate::is_scene_cut` |
| State | Last output + last source only (streamable; no full-clip buffer) |
| UI | Enable Stability for real: on/off + lock strength |

**Risks:** ghosting on fast motion, sticky detail across cuts, extra VRAM for previous full-res frame, interaction with color-match order.

**Acceptance:**

- ≥50% less flicker on static AI fabric/skin crops vs unlocked (simple residual metric + visual A/B)
- No worse ghosting on hard cuts than unlocked
- Export remains streaming (1–2 frames of state)

### 4.2 Quality — source restoration (P1)

**Partial today** (`src/media.rs`):

- Deinterlace: `bwdif` when broadcast profile + interlaced
- Restore: weak/strong deblock; strong + bilateral

**Still needed for old TV/movies:**

1. **IVTC / telecine** — detect 3:2 vs true interlaced **per scene** (DS9/Voyager note: effects shots break file-level cadence). Prefer inverse telecine for film pulldown; else deinterlace.
2. **Grain separation + re-injection** — hold high-frequency residual before the detail model; re-apply after temporal lock (scaled). Avoids plastic skin and grain shimmer amplification.
3. **Profile defaults** — AI video = minimal; DVD = deblock+IVTC; heavy compression = stronger restore.

### 4.3 Quality — anti-hallucination & recipes

Already strong: `clamp_deviation`, strength blend, resample-last, model guidance, Remacri default ranking.

**Further:**

- Content presets (AI / Film / Anime) setting model + strength + detail limit + restore
- Optional face ROI second pass (NMKD Faces) after global upscale
- Target-format scale (HD/2K/4K) using display aspect + pillarbox/crop for 4:3 (designed DECISIONS §7)
- Encode quality presets (CQ / HEVC vs H.264) so encode doesn’t re-block the upscale

### 4.4 Performance — bottlenecks

| # | Bottleneck | Est. gain | Notes |
|---|---|---|---|
| 1 | Heavy RRDB architecture | 5–20× with SPAN/Compact | Largest long-term |
| 2 | No TensorRT | 2–4× | TRT provider DLLs already appear under `target/*` but are unused |
| 3 | Serial tiles | 1.5–2× | One `run_tile` at a time in `Upscaler::run` |
| 4 | Host NCHW pack + H2D/D2H per tile | 1.2–1.5× | CPU build every tile |
| 5 | 4× model for 2× target | ~1.2× measured | Prefer native 2× model |
| 6 | Session create mixed into timings | — | Makes FP16 comparison invalid today |
| 7 | RIFE multiplies detail work | policy | Every output frame still upscaled |

**Note:** Performance work is **after** temporal lock so we accelerate the real product path (including lock cost).

### 4.5 Performance — recommended work (after lock)

1. **Instrumentation** — pure infer ms, tiles, lock/flow, post, encode; log lines in activity log  
2. **Tile batching** — batch dimension / uniform padded tiles; assembly stays serial  
3. **TensorRT EP** — optional, engine cache under AppData, CUDA fallback  
4. **Native scale match** — recommend/auto RealESRGAN_x2 for ×2  
5. **Session reuse / warm-up** — exclude load from ETA; re-measure FP16 properly  
6. **Decode ∥ infer ∥ encode** queues (2–3 frames) once lock is stable  
7. **SPAN / Compact / RealPLKSR** native importers — beat Aiarty speed sustainably  

### 4.6 Models & pipelines

| Recipe | Intent |
|---|---|
| AI Clean | Minimal restore → detail → **lock** |
| Film/DVD | IVTC/deint → deblock → grain hold → detail → **lock** → grain |
| Anime | Light deblock → AnimeSharp → **lock** |
| Face Hero | Global upscale → face ROI refine |
| Speed | Light native-scale model + TRT + **lock** |

**Out of native-exe scope for now:** SeedVR2/diffusion VSR in-process, AMD/Vulkan first-class, cloud APIs.

---

## 5. Roadmap (quality first)

| Phase | Work | Exit criteria |
|---|---|---|
| ~~**0**~~ | ~~Lightweight baseline~~ | **DONE** — stage timers in the activity log; `baseline_stage_timings` reference export; numbers in §2 |
| **1** | **Temporal lock** | Stability real in UI; flicker bar met; no cut regressions |
| **2** | IVTC + grain | Broadcast samples clean; grain not plastic or crawly |
| **3** | Batch tiles, native scale, FP16 re-bench | ≥1.5× same model **with lock on** |
| **4** | TensorRT + cache | Additional ≥2× cached TRT vs Phase 3 CUDA |
| **5** | SPAN/Compact + presets | Practical multi-minute Speed path; Quality holds still |
| **6** | Target format, encode presets, pipeline overlap | Daily-driver polish |

Opportunistic micro-opts during Phase 1 (e.g. don’t reload session every preview) are fine if they don’t delay lock. Full batching/TRT wait until Phases 3–4.

---

## 6. Scorecard (“as good as the pros”)

### Reference assets

- AI clean: `SpaceVoyage.mp4`, `SpaceVoyage_2.mp4` (see NEXT-SESSION)  
- Later: interlaced DVD, grainy film, anime still-motion, talking head  

### Metrics

| Metric | Method |
|---|---|
| Temporal flicker | Mean abs residual on static crops; flow-warped residual |
| Spatial quality | Blind A/B vs Topaz/Aiarty on same crop |
| Artifacts | Comb teeth, blocking, ink strokes on skin, ghosting |
| Speed | s/frame wall, pure infer, GPU util |
| Correctness | Existing tests: frames, fps, audio, even dims, SAR |

### Pass bars (proposed)

- **Temporal:** ≥50% flicker reduction on static AI crop without cut ghosting  
- **Speed (same model, post-lock):** ≥3× vs current 896px serial CUDA baseline after batching+TRT  
- **Broadcast:** no residual combing; correct film cadence when IVTC applies  
- **Still competitive:** locked Remacri/UltraSharp matches or beats Topaz “standard” models on clean AI sources in A/B  

---

## 7. Implementation touch map (when coding starts)

| Area | Primary files |
|---|---|
| Temporal lock | new `src/temporal.rs`; `export.rs`; `main.rs`; Stability in `ui/app.slint` |
| Batching / TRT | `upscale.rs`; `cuda.rs`; `Cargo.toml` (ort features); settings |
| Source prep | `media.rs`; Source section UI/settings |
| Grain | `media.rs` + post path in export/upscale |
| Models | `pth.rs`; `models/`; optional `models.json` metadata |
| Presets | `settings.rs`; UI |

Do not reopen closed bugslist items unless a new regression appears. Preserve RIFE order (interp before detail at source resolution) unless `RIFE-IMPLEMENTATION.md` is updated by design.

---

## 8. Decision checkpoints (open)

1. **Flow model:** RAFT-small ONNX vs lighter DIS-style estimator — pick smallest stable at 480p–1080p during Phase 1.  
2. **TRT packaging:** ship TensorRT runtime in-box vs require user install — in-box preferred for self-contained product (Phase 4).  
3. **Default recipe:** AI Clean vs Film/DVD for first-run UX.  
4. **Face stage:** include after lock, or defer to Phase 5+.  

---

## 9. Suggested first implementation goal

**Title:** Phase 0–1 — temporal lock (flow-guided detail stability)

**In scope:**

- Unlocked baseline exports + stage timers  
- Optical flow ONNX + warp + confidence + cut reset  
- Preview + export  
- Real Stability controls  
- Flicker scorecard on SpaceVoyage crops  

**Out of scope for that goal:** tile batching, TensorRT, SPAN, IVTC, grain (later phases).

**Longer arc:**

> Make Kraken video output temporally stable first, then fast enough for full scenes: Topaz-competitive quality on clean/AI sources, practical episode-length throughput on RTX 3090-class GPUs, with broadcast restoration complete enough for DVD-era TV.

---

## 10. Bottom line

| Priority | Work | Needle moved |
|---|---|---|
| **P0** | **Temporal lock** | Quality vs Topaz — **first** |
| P0 | Lightweight baseline timing | Prove lock; catch regressions |
| P1 | IVTC + grain | Broadcast quality |
| P1 | Tile batch + TRT + native scale | Speed vs Aiarty — **after lock** |
| P2 | SPAN/Compact + presets | Sustainable speed/quality |
| P2 | Face stage, target format, pipeline overlap | Pro polish |

Spatial quality is close. Pro parity is **temporal stability + proper source prep + then speed infrastructure**, in that order.
