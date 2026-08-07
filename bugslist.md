# Kraken Upscale — bug list

Static code audit against the current tree, informed by [DECISIONS.md](DECISIONS.md) and [NEXT-SESSION.md](NEXT-SESSION.md).

**Scope:** defects and misleading behaviour only. Planned temporal processing and source restoration stay in DECISIONS §13 unless they present as working UI that does nothing.

**Reviewed and corrected.** Findings below were independently checked against
the current Rust/Slint implementation and the earlier Grok design conversation.
Implementation status is updated as fixes land.

Last updated: 2026-08-06

---

## Completion checklist

- [x] B1 — close exits and signals active export cancellation
- [x] B2 / B9 / B25 — export cancellation, child cleanup, and partial-file removal
- [x] B3 — both preview pan modes use a stable pointer-down origin
- [x] B4 — export dimensions are rounded to valid even values
- [x] B5 — decoder and encoder diagnostics are continuously drained
- [x] B6 — beside-app `models/` discovery no longer requires an index file
- [x] B7 / B8 / B20 — queue scheduling skips finished/running jobs and blocks duplicates
- [x] B10 / B11 / B13 / B18 — misleading or inactive controls are honest and disabled
- [x] B12 — automatic tile size is real, persisted, and VRAM-aware
- [x] B14 — timing uses an explicit rational CFR policy
- [x] B15 — non-square-pixel sources are normalized to display geometry
- [x] B16 — inference discovers model input/output names
- [x] B17 — closed as not a bug; automated UI test confirms dock expansion
- [x] B19 — output paths are unique and the encoder refuses overwrite
- [x] B21 — unknown scale is labelled `auto`, then probed on load
- [x] B22 — machine-specific development runtime paths removed
- [x] B23 — closed as a non-bug after review
- [x] B24 — clean frame-boundary EOF and truncated-frame EOF are distinguished
- [x] B26 — Explorer file drops use the native window event stream and show hover feedback
- [x] B27 — missing GPU runtime no longer appears as an unexplained frozen preview
- [x] B28 — current video and export queue can be cleared independently
- [x] B29 — every selectable Source Restore filter runs in the bundled video engine
- [x] B30 — automatic tiles are capped safely for native 8x models

All confirmed defects in this audit are closed. Native folder-drop loading is
also complete for standard RRDBNet `.pth` checkpoints and `.onnx` models.

---

## Critical / high

### B1 — Window close does not quit the process
- **Where:** `src/main.rs` (`on_close_requested` → `CloseRequestResponse::HideWindow`)
- **What:** Closing the window hides it and leaves the event loop, preview worker, export worker, and GPU poller running. No tray or other quit path.
- **Impact:** Zombie process until killed externally.
- **Suggested fix:** Quit the event loop on close; signal background workers to stop.

### B2 — Export cancel is dead infrastructure
- **Where:** `src/export.rs` (`cancel` checks); `src/main.rs` (`spawn_exporter` creates `AtomicBool` never set true); `ui/app.slint` (no cancel control)
- **What:** Cancel is plumbed into the export loop but never armed from the UI. Matches DECISIONS §13 (“Export cancel button — Missing”).
- **Impact:** Long exports cannot be stopped from the app.
- **Suggested fix:** Shared cancel flag, UI button, kill decoder/encoder children, remove or quarantine partial output.
- **Audit note:** Factually correct, but this is documented unfinished functionality rather than a newly discovered regression. It remains high-priority because an export can run for hours.

### B3 — Preview pan compounds every move event
- **Where:** `ui/app.slint` (split view `sp` TouchArea; side-by-side `sbsa` TouchArea)
- **What:**
  ```slint
  root.pan-x += self.mouse-x - self.pressed-x;
  root.pan-y += self.mouse-y - self.pressed-y;
  ```
  `pressed-x/y` is the press origin, not the previous sample. Each `moved` event re-adds the full offset from press.
- **Impact:** Panning overshoots badly (drag feels exponential / unusable).
- **Suggested fix:** Store pan at pointer-down and set `pan = pan_at_press + (mouse - pressed)`, or accumulate deltas from last position.

### B4 — Odd output dimensions break yuv420p / NVENC
- **Where:** `src/export.rs` (`output_size`, encoder `-pix_fmt yuv420p`)
- **What:** Output size is `width * scale` × `height * scale` with no even-rounding. Odd source × odd scale can yield odd dimensions.
- **Impact:** Encode can fail on valid sources (e.g. some web resolutions ×3).
- **Suggested fix:** Force even width/height (round or crop) before encoding.

### B5 — Encoder stderr pipe can deadlock
- **Where:** `src/export.rs` (`spawn_encoder`: `stderr(Stdio::piped())`, read only after `wait()` on failure)
- **What:** If the encoder writes enough to fill the pipe, it blocks on write while the parent blocks on `wait`. Decoder correctly uses `stderr(Stdio::null())`.
- **Impact:** Export hang under chatty/error-heavy encoder output. The block can occur while writing frames as well as while waiting for the child.
- **Suggested fix:** Null stderr, or drain stderr on a side thread for the process lifetime.

### B6 — Packaged models directory requires `models.json`
- **Where:** `src/upscale.rs` (`models_dir`)
- **What:** Models beside the executable are only used if `models/models.json` exists; otherwise falls back to `CARGO_MANIFEST_DIR`, which is absent in a shipped install.
- **Impact:** Dropping only `.onnx` files next to a release binary fails discovery — contradicts DECISIONS §6 (“models/ is the source of truth”).
- **Suggested fix:** Prefer `exe/models` if the directory exists or contains any `.onnx`; treat `models.json` as optional metadata only.

---

## Medium

### B7 — Export Current always runs queue index 0
- **Where:** `src/main.rs` (`on_export_current` → `q.first()`)
- **What:** There is no selected queue row. The button label implies the current clip/settings, but it always starts the first queued snapshot and re-runs it even if already done.
- **Impact:** Wrong task exported; confusing UX for multi-item queues.

### B8 — Batch export re-runs finished tasks
- **Where:** `src/main.rs` (`on_batch_export`)
- **What:** Sends the entire queue with no skip of `done` / already-running rows.
- **Impact:** Accidental re-exports; wasted GPU time.

### B9 — Cancel path leaves partial output (when cancel is wired)
- **Where:** `src/export.rs` (`run` cancel branch)
- **What:** On cancel, the encoder input is closed and the encoder is waited cleanly, but the unwanted partial file is not deleted or quarantined.
- **Impact:** Partial `.mp4` left on disk; will matter once B2 is fixed.

### B10 — FP16 UI claim contradicts measured behaviour
- **Where:** `ui/app.slint` (Half precision CheckRow hint); DECISIONS §11a
- **What:** UI says “Roughly 2x faster on your 3090.” DECISIONS records FP16 as slower in the only measurement, and notes those timings included session creation so the comparison is invalid.
- **Impact:** Users enable FP16 expecting a speedup they may not get.

### B11 — Hardware dropdown is fake
- **Where:** `ui/app.slint` (Detail Model → FieldRow “Hardware”)
- **What:** Hardcoded `["NVIDIA GeForce RTX 3090", "CPU (fallback)"]` with no binding to backend provider selection.
- **Impact:** Looks like a real control; changing it does nothing.

### B12 — Auto tile size checkbox is inert
- **Where:** `ui/app.slint` (Detail Model); `src/main.rs` (`TILE = 256` constant)
- **What:** `checked: true` with no property binding. Tile size is always 256/16. Matches DECISIONS §13.
- **Impact:** Users believe VRAM-aware tiling is active.

### B13 — Large sections of the rail are UI-only
- **Where:** Source (content / restore / detect fields / preserve grain); Stability (lock / cut detect); Frame Interpolation; Color
- **What:** Values persist in settings but never enter the preview or export pipeline. Documented as open work in DECISIONS §13, but were presented as interactive controls without clear planned/unavailable labels.
- **Impact:** False confidence that preprocessing / temporal lock / RIFE are running.

### B14 — Frame rate probe uses `r_frame_rate` only
- **Where:** `src/media.rs` (`probe`)
- **What:** Uses stream `r_frame_rate` for fps and for encoder `-r`. No `avg_frame_rate`; frame count falls back to `duration * fps`.
- **Impact:** VFR or mis-tagged containers → wrong progress totals and possible A/V desync on encode. Reading `avg_frame_rate` alone is not a complete fix because the raw-frame pipe discards timestamps; the pipeline needs an explicit VFR-to-CFR policy or timestamp preservation.

### B15 — No SAR / display-aspect handling
- **Where:** `src/media.rs`, export sizing; designed in DECISIONS §7
- **What:** Coded pixel dimensions only; no `sample_aspect_ratio` / `display_aspect_ratio`.
- **Impact:** DVD / anamorphic sources export with squashed or stretched geometry if treated as square pixels.
- **Priority note:** High for this product: old TV/DVD material is a primary target, and the raw RGB pipe otherwise discards display geometry.

### B16 — ONNX I/O names hardcoded
- **Where:** `src/upscale.rs` (`run_tile`: `"input"` / `"output"`)
- **What:** Session run assumes those tensor names.
- **Impact:** Otherwise-valid graphs with different names fail at inference.

---

## Suspected (needs runtime confirmation)

### B17 — Left dock panel may not expand
- **Where:** `ui/app.slint` (dock `Rectangle` with animated `width` inside `HorizontalLayout`)
- **What:** LOGS / SYSTEM tabs render; panel expansion was never confirmed (DECISIONS §10, NEXT-SESSION “First thing to check”). Suspicion: layout overrides animated `width`.
- **Impact:** Dock appears broken if confirmed.
- **Suggested check:** Click SYSTEM / LOGS; if width stuck, drive `min-width` / `max-width` instead of `width`.
- **Audit note:** Not established statically. The explicit dynamic width is on a direct layout child and should be honored; keep this as runtime verification, not a confirmed defect.

---

## Low / polish

### B18 — Export Settings footer is non-functional
- **Where:** `ui/app.slint` (sticky export footer `TouchArea`)
- **What:** Gear + summary + chevron; click does nothing (no dialog, codec/CQ not user-editable).
- **Impact:** Looks clickable; codec stays hardcoded `h264_nvenc` / CQ 19.

### B19 — Add Task overwrites the same output path
- **Where:** `src/main.rs` (`on_add_task` path: `{stem}_{scale}x_{model}.mp4`)
- **What:** No uniqueness suffix; no overwrite warning. Strength / detail limit not in the name.
- **Impact:** Silent overwrite of previous export for same model+scale.
- **Priority note:** This is potential user data loss and should be treated as medium/high, not merely polish.

### B20 — No guard against double-queued batch
- **Where:** `src/main.rs` (export channel + queue)
- **What:** Batch/Export Current can be sent again while the exporter is still working the same indices.
- **Impact:** Duplicate work / confusing task row state.

### B21 — Drop-in models show `(0x)` until first load
- **Where:** `src/main.rs` (model name list); `src/upscale.rs` (`scale: 0` means probe on load)
- **What:** UI displays native scale before probe and never updates the registry label afterward, so an unlisted model stays `(0x)` for the entire session.
- **Impact:** Cosmetic confusion only.

### B22 — CUDA discovery hardcodes machine-specific paths
- **Where:** `src/cuda.rs` (`candidates`: fixed PyTorch venv paths under `F:\` / `E:\`)
- **What:** Fine for this development machine; wrong for any other install.
- **Resolution:** Removed. Discovery now uses only the beside-app runtime folder,
  the explicit app override, or a standard system toolkit location.

### B23 — Dock / view-mode only saved on close
- **Where:** `src/main.rs` (close handler); settings-changed intentionally skips dock/view
- **What:** Normal close saves these fields before returning `HideWindow`; B1 does not make persistence worse. Only a crash loses changes made since launch.
- **Audit verdict:** Not a meaningful bug. Saving these lightweight fields immediately would be nicer, but save-on-close is internally consistent.

---

## Additional verified findings

### B24 — Truncated decoder output can be reported as success
- **Where:** `src/export.rs` (`read_exact` handling)
- **What:** `read_exact` returns `UnexpectedEof` both for a clean zero-byte EOF at a frame boundary and for EOF after part of a frame. The loop treats both as normal completion despite the comment claiming it distinguishes them.
- **Impact:** A decoder failure can produce a shortened video that the app marks done.
- **Suggested fix:** Count bytes while filling each frame and treat EOF as clean only when zero bytes were read for the next frame.

### B25 — Export error paths do not guarantee child-process cleanup
- **Where:** `src/export.rs` (all `?` exits after decoder/encoder spawn)
- **What:** Inference, resample, or encoder-write errors can return before both ffmpeg children are explicitly killed/waited and before partial output is removed.
- **Impact:** Stray ffmpeg work, incomplete files, and loss of the useful encoder error.
- **Suggested fix:** Central cleanup/RAII guard covering decoder, encoder, stderr draining, and partial-file policy on every exit path.

### B26 — Drop target text is shown but Explorer drops are ignored
- **Where:** `src/dnd.rs`, `src/main.rs`, `ui/app.slint`
- **What:** The app installed a legacy `WM_DROPFILES` window-procedure hook while
  the window framework had already registered the modern OLE drop target.
  Explorer delivered the file through the framework, so the legacy hook never
  received it.
- **Resolution:** Listen to the framework's native `DroppedFile`, `HoveredFile`,
  and cancellation events. The empty stage now highlights while a valid file is
  over the window and clears that state on drop or cancellation.

### B27 — CPU fallback looks frozen and the activity log gives no progress
- **Where:** `src/main.rs` preview worker and runtime status
- **What:** A full-resolution RRDBNet preview could silently begin on the CPU
  when the packaged GPU runtime was absent. The status stayed `RUNTIME OK`, and
  the log stopped after model load while a very long render ran.
- **Resolution:** Missing GPU support is now a visible blocking state; automatic
  preview stops rather than starting impractical CPU work. GPU previews log
  dimensions, model loading, tile progress with elapsed time/ETA, completion,
  and failures. The same entries persist to `activity.log` across restarts.

### B28 — No way to clear the current video or export queue
- **Where:** `ui/app.slint`, `src/main.rs`
- **Resolution:** Added independent Clear Video and Clear Queue actions. Queue
  clearing requests cancellation when necessary; late preview results are
  generation-guarded so clearing/replacing a video cannot repaint stale output.

### B29 — Heavy-compression Restore generated an invalid filter chain
- **Where:** `src/media.rs`
- **What:** The strongest Restore choice used a denoiser absent from the bundled
  video engine, so selecting it failed before the preview could be decoded.
- **Resolution:** Replaced it with the bundled threaded bilateral denoiser,
  logged full source-processing failures to `activity.log`, and executed all
  four Restore values in the media test and against `SpaceVoyage_2.mp4`.

### B30 — Larger automatic tiles overflow native 8x CUDA graphs
- **Where:** `src/upscale.rs`, `src/main.rs`
- **Resolution:** Cap a tile by native model scale so its model output stays at
  or below 4096px. The RTX 3090 retains the measured 896px tile for 2x/4x
  models, while an 8x model is safely capped at 512px.

---

## Not bugs (known open work / by design)

These are intentional or documented unfinished work, not silent defects:

| Item | Notes |
|------|--------|
| Native `.pth` importer | Completed: direct ZIP and legacy checkpoint loading; 2×/4×/8× verified |
| Temporal lock / Stability pipeline | Designed; not implemented |
| Source preprocessing | Deinterlace/deblock/compression cleanup active; IVTC and grain remain |
| Scale by target format (HD/4K) | Designed in DECISIONS §7 |
| Performance (tile size, batching, TensorRT, model-scale match) | Known; not correctness bugs |
| Frame interpolation (RIFE) | UI only — listed as B13 when it misleads |
| LGPL ffmpeg → NVENC only, no libx264 | By design |
| AV1 NVENC fails on Ampere (3090) | Hardware limit, documented |

---

## Already in good shape (for regression awareness)

- End-to-end export validated (decode → model → encode → audio remux) on SpaceVoyage
- Model preference ranking (Remacri default, not alphabetical AnimeSharp)
- Scale applies via resample-last (native model scale → target)
- Settings persist model **by name**
- GPU telemetry polled off the UI thread
- Preview worker coalesces to newest job
- SectionHeader TouchArea declared under siblings (known Slint click-swallow trap fixed)
- Before/after divider tracks displayed image box, not the pane

---

## Verification record

1. Full Rust test suite passes, including exact/truncated frame reads, odd-size
   export sizing, anamorphic media normalization, and animated dock width.
2. Direct checkpoint import and inference pass with unchanged community models:
   2× RealESRGAN, 4× RealESRGAN, 4× UltraSharp, 4× Remacri, and 8× NMKD Faces.
3. Release compilation succeeds after the complete fix set.

---

## Audit notes

- Method: static review plus targeted automated and real-model runtime tests. Dock expansion (B17) is runtime-verified.
- Shared Grok conversation was reviewed for architectural context; bug verdicts are based on the current repository.
- Do **not** treat this list as a substitute for DECISIONS open work; link both when prioritising.
