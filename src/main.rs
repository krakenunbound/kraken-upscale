#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

use slint::{Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, VecModel};

mod cuda;
mod dnd;
mod export;
mod gpu;
mod media;
mod pth;
mod settings;
mod upscale;

slint::include_modules!();

/// Upscale factors offered by the Scale dropdown.
const SCALES: [u32; 3] = [2, 3, 4];

/// Preview tiling. Small enough to stay well inside VRAM at 4x, large enough
/// that per-tile overhead stays irrelevant.
const SAFE_TILE: u32 = 256;
const OVERLAP: u32 = 16;

fn recommended_tile() -> u32 {
    match gpu::stats().map(|s| s.vram_total_gb()) {
        Some(vram) if vram >= 20.0 => 512,
        Some(vram) if vram >= 10.0 => 384,
        _ => SAFE_TILE,
    }
}

fn to_image(frame: &media::Frame) -> Image {
    let mut buf = SharedPixelBuffer::<Rgb8Pixel>::new(frame.width, frame.height);
    buf.make_mut_bytes().copy_from_slice(&frame.rgb);
    Image::from_rgb8(buf)
}

/// A preview render request handed to the preview worker.
struct Job {
    clip: media::ClipInfo,
    at_secs: f64,
    model: upscale::ModelEntry,
    target_scale: u32,
    strength: f32,
    detail_limit: f32,
    fp16: bool,
    tile: u32,
}

fn main() -> Result<(), slint::PlatformError> {
    // Must happen before any ONNX Runtime session is created, which is before
    // the window exists — so this result is logged once the UI is up.
    let cuda_dir = cuda::prepare();

    let ui = MainWindow::new()?;
    ui.set_tasks(ModelRc::new(VecModel::<TaskRow>::from(Vec::new())));
    ui.set_log_lines(ModelRc::new(VecModel::<slint::SharedString>::from(
        Vec::new(),
    )));

    let logger = ui.as_weak();
    match cuda_dir.as_deref() {
        Some(dir) => log(&logger, format!("CUDA runtime: {}", dir.display())),
        None => log(
            &logger,
            "CUDA runtime not found — inference will run on CPU",
        ),
    }

    let probe = gpu::probe();
    let auto_tile_size = recommended_tile();
    ui.set_gpu_name(probe.name.clone().into());
    ui.set_runtime_state(probe.runtime.into());
    log(&logger, format!("GPU: {}", probe.name));
    log(&logger, format!("automatic tile size: {auto_tile_size}px"));

    let rt = gpu::runtime(cuda_dir.as_deref());
    ui.set_rt_app(rt.app.into());
    ui.set_rt_ort(rt.ort.into());
    ui.set_rt_onnx(rt.onnxruntime.into());
    ui.set_rt_ffmpeg(rt.ffmpeg.clone().into());
    ui.set_rt_cuda(rt.cuda.clone().into());
    ui.set_rt_cudnn(rt.cudnn.clone().into());
    ui.set_rt_provider(if cuda_dir.is_some() { "CUDA" } else { "CPU" }.into());
    ui.set_models_dir(upscale::models_dir().display().to_string().into());
    log(
        &logger,
        format!(
            "runtime: ffmpeg {} · cuda {} · cudnn {}",
            rt.ffmpeg, rt.cuda, rt.cudnn
        ),
    );

    spawn_gpu_monitor(ui.as_weak());

    let models = upscale::registry();
    if models.is_empty() {
        ui.set_runtime_state("NO MODELS FOUND".into());
        log(&logger, "no models found — drop .onnx files into models/");
    } else {
        ui.set_models_count(models.len() as i32);
        log(
            &logger,
            format!(
                "{} models: {}",
                models.len(),
                models
                    .iter()
                    .map(|m| m.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    }
    ui.set_model_names(ModelRc::new(VecModel::from(
        models
            .iter()
            .map(|m| {
                let scale = if m.scale == 0 {
                    "auto".to_string()
                } else {
                    format!("{}x", m.scale)
                };
                slint::SharedString::from(format!("{}  ({scale})", m.name))
            })
            .collect::<Vec<_>>(),
    )));

    // Restore the previous session. The model is stored by name, so it
    // survives files being added to or removed from the models folder;
    // alphabetical order would otherwise put AnimeSharp first, the worst
    // possible default for photographic footage.
    let saved = settings::load();
    if !models.is_empty() {
        let idx = models
            .iter()
            .position(|m| m.name == saved.model)
            .unwrap_or_else(|| upscale::preferred_default(&models));
        ui.set_model_index(idx as i32);
        ui.set_model_note(upscale::guidance(&models[idx].name).into());
    }
    ui.set_scale_index(saved.scale_index);
    ui.set_strength(saved.strength);
    ui.set_detail_limit(saved.detail_limit);
    ui.set_fp16(saved.fp16);
    ui.set_auto_tile(saved.auto_tile);
    ui.set_content_index(saved.content_index);
    ui.set_restore_index(saved.restore_index);
    ui.set_detect_fields(saved.detect_fields);
    ui.set_preserve_grain(saved.preserve_grain);
    ui.set_stability_on(saved.stability);
    ui.set_lock(saved.lock);
    ui.set_cut_detect(saved.cut_detect);
    ui.set_interp_on(saved.interp_enabled);
    ui.set_interp_index(saved.interp_index);
    ui.set_dock_panel(saved.dock_panel);
    ui.set_view_mode(saved.view_mode);
    log(&logger, format!("settings: {}", settings::path().display()));

    let clip: Rc<RefCell<Option<media::ClipInfo>>> = Rc::new(RefCell::new(None));
    let tx = spawn_preview_worker(ui.as_weak());

    // Kick a preview render from whatever the controls currently say.
    let request = {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let models = models.clone();
        let tx = tx.clone();
        move || {
            let ui = handle.unwrap();
            let Some(info) = clip.borrow().clone() else {
                return;
            };
            let Some(model) = models.get(ui.get_model_index().max(0) as usize).cloned() else {
                return;
            };
            ui.set_model_note(upscale::guidance(&model.name).into());
            let target_scale = SCALES[(ui.get_scale_index().max(0) as usize).min(2)];

            let at_secs = ui.get_playhead().clamp(0.0, 1.0) as f64 * info.duration;
            ui.set_out_badge(
                format!(
                    "{} × {} · {}",
                    info.width * target_scale,
                    info.height * target_scale,
                    model.name
                )
                .into(),
            );
            ui.set_preview_status("upscaling sample frame…".into());

            let _ = tx.send(Job {
                clip: info,
                at_secs,
                model,
                target_scale,
                strength: ui.get_strength(),
                detail_limit: ui.get_detail_limit(),
                fp16: ui.get_fp16(),
                tile: if ui.get_auto_tile() {
                    auto_tile_size
                } else {
                    SAFE_TILE
                },
            });
        }
    };

    let open = {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let request = request.clone();
        move |path: std::path::PathBuf| {
            let Some(ui) = handle.upgrade() else { return };
            match media::probe(&path) {
                Ok(info) => {
                    log(
                        &ui.as_weak(),
                        format!(
                            "opened {} · {}×{} · {:.2} fps · {} frames · {} / {} · fields: {}",
                            info.file_name(),
                            info.width,
                            info.height,
                            info.fps,
                            info.frames,
                            info.codec,
                            info.pix_fmt,
                            info.field_order
                        ),
                    );
                    load_clip(&ui, &info);
                    *clip.borrow_mut() = Some(info);
                    request();
                }
                Err(e) => {
                    log(
                        &ui.as_weak(),
                        format!("could not read {}: {e}", path.display()),
                    );
                    ui.set_preview_status(format!("could not read file: {e}").into());
                }
            }
        }
    };

    {
        let open = open.clone();
        ui.on_import_clicked(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Video", &["mp4", "mkv", "mov", "avi", "webm", "m2ts", "ts"])
                .add_filter("All files", &["*"])
                .pick_file()
            {
                open(path);
            }
        });
    }

    {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let request = request.clone();
        ui.on_seek(move |pos| {
            let ui = handle.unwrap();
            if let Some(info) = clip.borrow().as_ref() {
                // Show the source frame immediately; the model result follows.
                if let Ok(f) =
                    media::frame_at(info, pos.clamp(0.0, 1.0) as f64 * info.duration, None)
                {
                    ui.set_frame_before(to_image(&f));
                }
            }
            request();
        });
    }

    // Every control funnels through here, so persistence hangs off the same
    // signal that triggers a re-render.
    {
        let handle = ui.as_weak();
        let models = models.clone();
        let request = request.clone();
        ui.on_settings_changed(move || {
            if let Some(ui) = handle.upgrade() {
                settings::save(&collect_settings(&ui, &models));
            }
            request();
        });
    }

    ui.on_reveal(|path| reveal(&path));

    {
        let handle = ui.as_weak();
        ui.on_reset_view(move || {
            let ui = handle.unwrap();
            ui.set_zoom(1.0);
            ui.set_pan_x(0.0);
            ui.set_pan_y(0.0);
        });
    }

    // ── Export queue ────────────────────────────────────────────────────
    let queue: Rc<RefCell<Vec<export::ExportSpec>>> = Rc::new(RefCell::new(Vec::new()));
    let active_cancel: Arc<Mutex<Option<Arc<AtomicBool>>>> = Arc::new(Mutex::new(None));
    let export_tx = spawn_exporter(ui.as_weak(), active_cancel.clone());

    {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let models = models.clone();
        let queue = queue.clone();
        ui.on_add_task(move || {
            let ui = handle.unwrap();
            let Some(info) = clip.borrow().clone() else {
                return;
            };
            let Some(model) = models.get(ui.get_model_index().max(0) as usize).cloned() else {
                return;
            };
            let target_scale = SCALES[(ui.get_scale_index().max(0) as usize).min(2)];
            let strength = ui.get_strength();
            let detail_limit = ui.get_detail_limit();

            let stem = info
                .path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".into());
            let candidate = info
                .path
                .with_file_name(format!("{stem}_{target_scale}x_{}.mp4", model.name));
            let out_path = unique_output_path(&candidate, &queue.borrow());

            let spec = export::ExportSpec {
                clip: info.clone(),
                model: model.clone(),
                target_scale,
                strength,
                detail_limit,
                fp16: ui.get_fp16(),
                tile: if ui.get_auto_tile() {
                    auto_tile_size
                } else {
                    SAFE_TILE
                },
                out_path: out_path.clone(),
                codec: "h264_nvenc".into(),
                cq: 19,
            };

            let (ow, oh) = spec.output_size();
            let row = TaskRow {
                name: info.file_name().into(),
                detail: format!(
                    "{ow}×{oh} · {} · str {strength:.2} · lim {detail_limit:.2} · {} frames",
                    model.name, info.frames
                )
                .into(),
                status: "queued".into(),
                path: out_path.display().to_string().into(),
                progress: 0.0,
                running: false,
                failed: false,
                done: false,
            };

            queue.borrow_mut().push(spec);
            if let Some(vm) = ui.get_tasks().as_any().downcast_ref::<VecModel<TaskRow>>() {
                vm.push(row);
            }
        });
    }

    {
        let handle = ui.as_weak();
        let queue = queue.clone();
        let tx = export_tx.clone();
        ui.on_export_current(move || {
            let ui = handle.unwrap();
            let q = queue.borrow();
            let next = ui
                .get_tasks()
                .as_any()
                .downcast_ref::<VecModel<TaskRow>>()
                .and_then(|vm| {
                    (0..vm.row_count()).find(|&idx| {
                        vm.row_data(idx)
                            .map(|row| !row.done && !row.running)
                            .unwrap_or(false)
                    })
                });
            match next.and_then(|idx| q.get(idx).cloned().map(|spec| (idx, spec))) {
                Some((idx, spec)) => {
                    mark_scheduled(&ui, idx);
                    let _ = tx.send(vec![(idx, spec)]);
                }
                None => ui.set_preview_status("no unfinished task is available".into()),
            }
        });
    }

    {
        let handle = ui.as_weak();
        let queue = queue.clone();
        let tx = export_tx.clone();
        ui.on_batch_export(move || {
            let ui = handle.unwrap();
            let q = queue.borrow();
            let batch: Vec<_> = ui
                .get_tasks()
                .as_any()
                .downcast_ref::<VecModel<TaskRow>>()
                .map(|vm| {
                    q.iter()
                        .cloned()
                        .enumerate()
                        .filter(|(idx, _)| {
                            vm.row_data(*idx)
                                .map(|row| !row.done && !row.running)
                                .unwrap_or(false)
                        })
                        .collect()
                })
                .unwrap_or_default();
            if batch.is_empty() {
                ui.set_preview_status("no unfinished tasks are available".into());
                return;
            }
            for (idx, _) in &batch {
                mark_scheduled(&ui, *idx);
            }
            let _ = tx.send(batch);
        });
    }

    {
        let active_cancel = active_cancel.clone();
        ui.on_cancel_export(move || {
            if let Ok(slot) = active_cancel.lock() {
                if let Some(cancel) = slot.as_ref() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        });
    }

    // Opening with a path argument covers "drag a file onto the exe".
    if let Some(arg) = std::env::args_os().nth(1) {
        open(std::path::PathBuf::from(arg));
    }

    // Dock and view-mode toggles deliberately don't trigger a re-render, so
    // they're captured here rather than through `settings-changed`.
    {
        let handle = ui.as_weak();
        let models = models.clone();
        let active_cancel = active_cancel.clone();
        ui.window().on_close_requested(move || {
            if let Ok(slot) = active_cancel.lock() {
                if let Some(cancel) = slot.as_ref() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            if let Some(ui) = handle.upgrade() {
                settings::save(&collect_settings(&ui, &models));
            }
            let _ = slint::quit_event_loop();
            slint::CloseRequestResponse::HideWindow
        });
    }

    ui.show()?;

    // The native window does not exist until the event loop has spun at least
    // once, so `show()` is too early to ask for an HWND. Defer by one tick.
    let dnd_timer = slint::Timer::default();
    {
        let handle = ui.as_weak();
        let open = open.clone();
        dnd_timer.start(
            slint::TimerMode::SingleShot,
            std::time::Duration::from_millis(80),
            move || {
                let Some(ui) = handle.upgrade() else { return };
                let open = open.clone();
                let w = ui.as_weak();
                match dnd::attach(ui.window(), open) {
                    Ok(()) => log(&w, "drag and drop ready"),
                    Err(e) => log(&w, format!("drag and drop unavailable: {e}")),
                }
            },
        );
    }

    slint::run_event_loop()?;
    Ok(())
}

/// Runs preview inference off the UI thread. Sessions are expensive to build,
/// so the loaded model is kept until a different one is asked for.
fn spawn_preview_worker(handle: slint::Weak<MainWindow>) -> Sender<Job> {
    let (tx, rx) = channel::<Job>();

    std::thread::spawn(move || {
        let mut loaded: Option<(String, bool, upscale::Upscaler)> = None;

        while let Ok(job) = rx.recv() {
            // Drain to the newest request — scrubbing and dragging sliders
            // queues jobs faster than they can be served, and only the last
            // one is worth rendering.
            let job = std::iter::successors(Some(job), |_| rx.try_recv().ok())
                .last()
                .unwrap();

            let started = std::time::Instant::now();

            let source = match media::frame_at(&job.clip, job.at_secs, None) {
                Ok(f) => f,
                Err(e) => {
                    report(&handle, format!("decode failed: {e}"), None);
                    continue;
                }
            };

            let stale = loaded
                .as_ref()
                .map(|(n, f, _)| n != &job.model.name || *f != job.fp16)
                .unwrap_or(true);
            if stale {
                match upscale::Upscaler::load(&job.model, job.fp16) {
                    Ok(u) => {
                        log(
                            &handle,
                            format!(
                                "loaded {} · {}x native · {} · {}",
                                job.model.name,
                                u.scale,
                                if job.fp16 { "fp16" } else { "fp32" },
                                u.provider
                            ),
                        );
                        // Report what ORT actually registered, not what was
                        // requested — CUDA can silently fall back.
                        let h = handle.clone();
                        let p = u.provider.clone();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = h.upgrade() {
                                ui.set_rt_provider(p.into());
                            }
                        });
                        loaded = Some((job.model.name.clone(), job.fp16, u));
                    }
                    Err(e) => {
                        log(&handle, format!("model load failed: {e}"));
                        report(&handle, format!("model load failed: {e}"), None);
                        continue;
                    }
                }
            }

            let Some((_, _, up)) = loaded.as_mut() else {
                continue;
            };

            // Feed tile progress back so the status bar fills as the frame
            // resolves rather than sitting blank for several seconds.
            let progress_handle = handle.clone();
            let mut on_progress = move |done: u32, total: u32| {
                let frac = if total == 0 {
                    0.0
                } else {
                    done as f32 / total as f32
                };
                let h = progress_handle.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = h.upgrade() {
                        ui.set_preview_progress(frac);
                    }
                });
            };

            match up.run(&source, job.tile, OVERLAP, &mut on_progress) {
                Ok(out) => {
                    // Order matters: clamp structural error first, then mix
                    // toward the source, then resample to the requested scale.
                    // Resampling last means a 4x model asked for 2x gets its
                    // invented high-frequency detail averaged away.
                    let limited = upscale::clamp_deviation(&out, &source, job.detail_limit * 255.0);
                    let blended = upscale::blend_toward_source(&limited, &source, job.strength);
                    let final_frame = upscale::resample(
                        &blended,
                        source.width * job.target_scale,
                        source.height * job.target_scale,
                    );

                    let secs = started.elapsed().as_secs_f32();
                    let mut notes = Vec::new();
                    if job.strength < 0.999 {
                        notes.push(format!("strength {:.2}", job.strength));
                    }
                    if job.detail_limit < 0.999 {
                        notes.push(format!("limit ±{:.0}", job.detail_limit * 255.0));
                    }
                    let extra = if notes.is_empty() {
                        String::new()
                    } else {
                        format!(" · {}", notes.join(" · "))
                    };

                    let note = format!(
                        "{} · {}x native → {}x{} · {} · {:.1}s on {}",
                        job.model.name,
                        up.scale,
                        job.target_scale,
                        extra,
                        if job.fp16 { "fp16" } else { "fp32" },
                        secs,
                        up.provider
                    );
                    report(&handle, note, Some(final_frame));
                }
                Err(e) => report(&handle, format!("inference failed: {e}"), None),
            }
        }
    });

    tx
}

/// Runs full-clip exports off the UI thread.
fn spawn_exporter(
    handle: slint::Weak<MainWindow>,
    active_cancel: Arc<Mutex<Option<Arc<AtomicBool>>>>,
) -> Sender<Vec<(usize, export::ExportSpec)>> {
    let (tx, rx) = channel::<Vec<(usize, export::ExportSpec)>>();

    std::thread::spawn(move || {
        let mut loaded: Option<(String, bool, upscale::Upscaler)> = None;

        while let Ok(batch) = rx.recv() {
            let cancel = Arc::new(AtomicBool::new(false));
            if let Ok(mut slot) = active_cancel.lock() {
                *slot = Some(cancel.clone());
            }
            set_export_running(&handle, true);

            for (idx, spec) in batch {
                if cancel.load(Ordering::Relaxed) {
                    set_task(&handle, idx, 0.0, "cancelled", false, false);
                    continue;
                }
                set_task(&handle, idx, 0.0, "loading model", true, false);

                let stale = loaded
                    .as_ref()
                    .map(|(n, f, _)| n != &spec.model.name || *f != spec.fp16)
                    .unwrap_or(true);
                if stale {
                    match upscale::Upscaler::load(&spec.model, spec.fp16) {
                        Ok(u) => {
                            log(
                                &handle,
                                format!(
                                    "export: loaded {} ({}x native) on {}",
                                    spec.model.name, u.scale, u.provider
                                ),
                            );
                            loaded = Some((spec.model.name.clone(), spec.fp16, u));
                        }
                        Err(e) => {
                            log(&handle, format!("export: model load failed: {e}"));
                            set_task(&handle, idx, 0.0, "model failed", false, true);
                            continue;
                        }
                    }
                }
                let Some((_, _, up)) = loaded.as_mut() else {
                    continue;
                };

                let started = std::time::Instant::now();
                let progress_handle = handle.clone();
                let mut on_progress = move |done: u64, total: u64| {
                    let frac = if total == 0 {
                        0.0
                    } else {
                        done as f32 / total as f32
                    };
                    // ETA from the running average. Per-frame cost is uniform
                    // enough here for that to be trustworthy after a frame or
                    // two — the first frame is skipped because it carries
                    // session creation and cuDNN autotuning.
                    let label = if done >= 2 && done < total {
                        let per = started.elapsed().as_secs_f64() / done as f64;
                        format!(
                            "{done}/{total} · {} left",
                            human_duration(per * (total - done) as f64)
                        )
                    } else {
                        format!("{done}/{total}")
                    };
                    let h = progress_handle.clone();
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(ui) = h.upgrade() {
                            if let Some(vm) =
                                ui.get_tasks().as_any().downcast_ref::<VecModel<TaskRow>>()
                            {
                                if let Some(mut row) = vm.row_data(idx) {
                                    row.progress = frac;
                                    row.status = label.into();
                                    row.running = true;
                                    vm.set_row_data(idx, row);
                                }
                            }
                        }
                    });
                };

                match export::run(
                    &spec,
                    up,
                    spec.tile,
                    OVERLAP,
                    cancel.clone(),
                    &mut on_progress,
                ) {
                    Ok(()) => {
                        let mins = started.elapsed().as_secs_f32() / 60.0;
                        set_task(
                            &handle,
                            idx,
                            1.0,
                            &format!("done · {mins:.1} min"),
                            false,
                            false,
                        );
                        log(
                            &handle,
                            format!("exported {} in {mins:.1} min", spec.out_path.display()),
                        );
                    }
                    Err(e) if e == "cancelled" => {
                        log(
                            &handle,
                            format!("export cancelled: {}", spec.out_path.display()),
                        );
                        set_task(&handle, idx, 0.0, "cancelled", false, false);
                    }
                    Err(e) => {
                        log(&handle, format!("export failed: {e}"));
                        set_task(&handle, idx, 0.0, "failed", false, true);
                    }
                }
            }

            if let Ok(mut slot) = active_cancel.lock() {
                *slot = None;
            }
            set_export_running(&handle, false);
        }
    });

    tx
}

/// Polls GPU telemetry on a background thread.
///
/// nvidia-smi takes ~50–100ms per call, which would be a visible stutter if it
/// ran on the UI thread from a timer.
fn spawn_gpu_monitor(handle: slint::Weak<MainWindow>) {
    std::thread::spawn(move || loop {
        if let Some(s) = gpu::stats() {
            let h = handle.clone();
            let posted = slint::invoke_from_event_loop(move || {
                let Some(ui) = h.upgrade() else { return };
                ui.set_sys_device(gpu::short_name(&s.device).into());
                ui.set_sys_usage(s.usage_pct as i32);
                ui.set_sys_temp(s.temp_c as i32);
                ui.set_sys_clock(s.clock_mhz as i32);
                ui.set_sys_clock_max(s.clock_max_mhz as i32);
                ui.set_sys_power(s.power_w);
                ui.set_sys_power_max(s.power_max_w);
                ui.set_sys_vram_used(s.vram_used_gb());
                ui.set_sys_vram_total(s.vram_total_gb());
                ui.set_sys_vram_free(s.vram_free_gb());
                ui.set_sys_driver(s.driver.clone().into());
            });
            // The event loop is gone once the window closes; stop polling
            // rather than spinning a detached thread for the process lifetime.
            if posted.is_err() {
                return;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(1500));
    });
}

/// Newest entries are kept at the top of the panel, so this bounds how far
/// back the log goes rather than how far forward.
const LOG_CAPACITY: usize = 400;

/// Appends a line to the activity log and mirrors it to stderr.
///
/// Safe to call from any thread — the model touch is marshalled onto the UI
/// thread.
fn log(handle: &slint::Weak<MainWindow>, msg: impl Into<String>) {
    let msg = msg.into();
    eprintln!("{msg}");

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| {
            let s = d.as_secs() % 86_400;
            format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
        })
        .unwrap_or_default();

    let handle = handle.clone();
    let _ = slint::invoke_from_event_loop(move || {
        let Some(ui) = handle.upgrade() else { return };
        if let Some(vm) = ui
            .get_log_lines()
            .as_any()
            .downcast_ref::<VecModel<slint::SharedString>>()
        {
            vm.insert(0, format!("{stamp}  {msg}").into());
            while vm.row_count() > LOG_CAPACITY {
                vm.remove(vm.row_count() - 1);
            }
        }
    });
}

/// "45s", "12m 30s", "1h 04m"
fn human_duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m {:02}s", s / 60, s % 60),
        _ => format!("{}h {:02}m", s / 3600, (s % 3600) / 60),
    }
}

fn unique_output_path(
    candidate: &std::path::Path,
    queue: &[export::ExportSpec],
) -> std::path::PathBuf {
    let reserved = |p: &std::path::Path| p.exists() || queue.iter().any(|item| item.out_path == p);
    if !reserved(candidate) {
        return candidate.to_path_buf();
    }

    let parent = candidate
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let stem = candidate
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".into());
    let ext = candidate
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4");
    for suffix in 2u32.. {
        let next = parent.join(format!("{stem}_{suffix}.{ext}"));
        if !reserved(&next) {
            return next;
        }
    }
    unreachable!()
}

fn set_task(
    handle: &slint::Weak<MainWindow>,
    idx: usize,
    progress: f32,
    status: &str,
    running: bool,
    failed: bool,
) {
    let handle = handle.clone();
    let status = status.to_string();
    let _ = slint::invoke_from_event_loop(move || {
        let Some(ui) = handle.upgrade() else { return };
        if let Some(vm) = ui.get_tasks().as_any().downcast_ref::<VecModel<TaskRow>>() {
            if let Some(mut row) = vm.row_data(idx) {
                row.progress = progress;
                row.status = status.into();
                row.running = running;
                row.failed = failed;
                // Reveal is only offered once the file exists.
                row.done = !running && !failed && progress >= 1.0;
                vm.set_row_data(idx, row);
            }
        }
    });
}

fn mark_scheduled(ui: &MainWindow, idx: usize) {
    if let Some(vm) = ui.get_tasks().as_any().downcast_ref::<VecModel<TaskRow>>() {
        if let Some(mut row) = vm.row_data(idx) {
            row.status = "scheduled".into();
            row.running = true;
            row.failed = false;
            vm.set_row_data(idx, row);
        }
    }
}

fn set_export_running(handle: &slint::Weak<MainWindow>, running: bool) {
    let handle = handle.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(ui) = handle.upgrade() {
            ui.set_export_running(running);
        }
    });
}

/// Snapshots the current control state for persistence.
fn collect_settings(ui: &MainWindow, models: &[upscale::ModelEntry]) -> settings::Settings {
    settings::Settings {
        model: models
            .get(ui.get_model_index().max(0) as usize)
            .map(|m| m.name.clone())
            .unwrap_or_default(),
        scale_index: ui.get_scale_index(),
        strength: ui.get_strength(),
        detail_limit: ui.get_detail_limit(),
        fp16: ui.get_fp16(),
        auto_tile: ui.get_auto_tile(),
        content_index: ui.get_content_index(),
        restore_index: ui.get_restore_index(),
        detect_fields: ui.get_detect_fields(),
        preserve_grain: ui.get_preserve_grain(),
        stability: ui.get_stability_on(),
        lock: ui.get_lock(),
        cut_detect: ui.get_cut_detect(),
        interp_enabled: ui.get_interp_on(),
        interp_index: ui.get_interp_index(),
        dock_panel: ui.get_dock_panel(),
        view_mode: ui.get_view_mode(),
        codec: "h264_nvenc".into(),
        cq: 19,
    }
}

/// Opens Explorer with the file selected.
fn reveal(path: &str) {
    let p = std::path::Path::new(path);
    if !p.exists() {
        eprintln!("reveal: {path} does not exist");
        return;
    }
    // Explorer exits non-zero even on success, so the status is ignored.
    let mut cmd = std::process::Command::new("explorer.exe");
    cmd.arg(format!("/select,{}", p.display()));
    let _ = cmd.spawn();
}

fn report(handle: &slint::Weak<MainWindow>, status: String, frame: Option<media::Frame>) {
    let handle = handle.clone();
    let _ = slint::invoke_from_event_loop(move || {
        let Some(ui) = handle.upgrade() else { return };
        ui.set_preview_status(status.into());
        ui.set_preview_progress(0.0);
        if let Some(f) = frame {
            ui.set_frame_after(to_image(&f));
        }
    });
}

fn load_clip(ui: &MainWindow, info: &media::ClipInfo) {
    ui.set_clip_name(info.file_name().into());
    ui.set_clip_specs(info.specs().into());
    let geometry = if info.coded_width != info.width || info.coded_height != info.height {
        format!(
            "{} × {} coded → {} × {} display",
            info.coded_width, info.coded_height, info.width, info.height
        )
    } else {
        format!("{} × {}", info.width, info.height)
    };
    ui.set_src_badge(format!("{geometry} · {}", info.codec).into());
    ui.set_clip_aspect(info.width as f32 / info.height.max(1) as f32);
    ui.set_clip_duration(info.duration as f32);
    ui.set_clip_fps(info.fps as f32);

    // Interlaced sources need the front-end before any detail model runs, so
    // surface it as soon as the clip loads rather than at export time.
    let mut notes = Vec::new();
    if info.is_interlaced() {
        notes.push(format!(
            "Interlaced ({}) — preprocessing is not active yet",
            info.field_order
        ));
    }
    if info.sample_aspect_ratio != "1:1" && info.sample_aspect_ratio != "1/1" {
        notes.push(format!(
            "SAR {} normalized to square pixels",
            info.sample_aspect_ratio
        ));
    }
    ui.set_source_note(notes.join(" · ").into());

    ui.set_playhead(0.0);
    ui.set_zoom(1.0);
    ui.set_pan_x(0.0);
    ui.set_pan_y(0.0);
    ui.set_has_clip(true);

    if let Ok(f) = media::frame_at(info, 0.0, None) {
        let img = to_image(&f);
        ui.set_frame_before(img.clone());
        ui.set_frame_after(img.clone());
        ui.set_thumb(img);
    }
}

#[cfg(test)]
mod ui_tests {
    use super::*;

    #[test]
    fn dock_width_tracks_open_state() {
        let ui = MainWindow::new().expect("create UI");
        assert_eq!(ui.get_dock_current_width(), 30.0);
        ui.set_dock_panel(2);
        std::thread::sleep(std::time::Duration::from_millis(200));
        slint::platform::update_timers_and_animations();
        assert_eq!(ui.get_dock_current_width(), 410.0);
        ui.set_dock_panel(0);
        std::thread::sleep(std::time::Duration::from_millis(200));
        slint::platform::update_timers_and_animations();
        assert_eq!(ui.get_dock_current_width(), 30.0);
    }
}
