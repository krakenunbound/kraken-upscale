#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::cell::{Cell, RefCell};
use std::io::Write;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

use slint::{Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, VecModel};

mod cuda;
mod dnd;
mod export;
mod gpu;
mod image_mode;
mod interpolate;
mod media;
mod pth;
mod settings;
mod stages;
mod temporal;
mod upscale;
mod video;

slint::include_modules!();

/// Upscale factors offered by the Scale dropdown.
const SCALES: [u32; 3] = [2, 3, 4];

/// Preview tiling. Small enough to stay well inside VRAM at 4x, large enough
/// that per-tile overhead stays irrelevant.
const SAFE_TILE: u32 = 256;
const OVERLAP: u32 = 16;

fn recommended_tile() -> u32 {
    match gpu::stats().map(|s| s.vram_total_gb()) {
        // Measured on the target RTX 3090: 896px was 44% faster than 512px
        // for the representative 704x1248 UltraSharp frame, while 1024px
        // regressed slightly.
        Some(vram) if vram >= 20.0 => 896,
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
#[derive(Clone)]
struct Job {
    generation: u64,
    clip: media::ClipInfo,
    at_secs: f64,
    model: upscale::ModelEntry,
    target_scale: u32,
    strength: f32,
    detail_limit: f32,
    source_treatment: media::SourceTreatment,
    match_color: bool,
    interpolation_fps: Option<u32>,
    fp16: bool,
    tile: u32,
    /// When set, the preview renders the preceding frame too and shows the
    /// locked result, so Stability can actually be judged before committing to
    /// an export.
    stability: bool,
    lock_strength: f32,
    cut_detect: f32,
}

#[derive(Clone)]
struct ImageJob {
    generation: u64,
    image: image_mode::ImageInfo,
    model: upscale::ModelEntry,
    target_scale: u32,
    strength: f32,
    detail_limit: f32,
    match_color: bool,
    fp16: bool,
    tile: u32,
}

#[derive(Clone)]
enum TaskSpec {
    Video(export::ExportSpec),
    Image(image_mode::ExportSpec),
}

impl TaskSpec {
    fn model(&self) -> &upscale::ModelEntry {
        match self {
            Self::Video(spec) => &spec.model,
            Self::Image(spec) => &spec.model,
        }
    }

    fn fp16(&self) -> bool {
        match self {
            Self::Video(spec) => spec.fp16,
            Self::Image(spec) => spec.fp16,
        }
    }

    fn out_path(&self) -> &std::path::Path {
        match self {
            Self::Video(spec) => &spec.out_path,
            Self::Image(spec) => &spec.out_path,
        }
    }
}

fn model_labels(models: &[upscale::ModelEntry]) -> ModelRc<slint::SharedString> {
    ModelRc::new(VecModel::from(
        models
            .iter()
            .map(|model| {
                let scale = if model.scale == 0 {
                    "auto".to_string()
                } else {
                    format!("{}x", model.scale)
                };
                slint::SharedString::from(format!("{}  ({scale})", model.display_name()))
            })
            .collect::<Vec<_>>(),
    ))
}

fn set_model_picker(ui: &MainWindow, models: &[upscale::ModelEntry], selected: &str) {
    ui.set_model_names(model_labels(models));
    let index = models
        .iter()
        .position(|model| model.name == selected)
        .unwrap_or_else(|| upscale::preferred_default(models));
    ui.set_model_index(index as i32);
    if let Some(model) = models.get(index) {
        ui.set_model_note(upscale::guidance(&model.name).into());
        ui.set_model_is_video(model.is_video());
        ui.set_model_video_tuned(model.video_tuned);
    } else {
        ui.set_model_note("No compatible models are installed for this mode.".into());
        ui.set_model_is_video(false);
        ui.set_model_video_tuned(false);
    }
}

fn models_for_mode(all: &[upscale::ModelEntry], mode: i32) -> Vec<upscale::ModelEntry> {
    all.iter()
        .filter(|model| {
            if mode == 1 {
                !model.is_video() && !model.video_tuned
            } else {
                model.is_video() || model.video_tuned
            }
        })
        .cloned()
        .collect()
}

fn main() -> Result<(), slint::PlatformError> {
    // Must happen before any ONNX Runtime session is created, which is before
    // the window exists — so this result is logged once the UI is up.
    let cuda_dir = cuda::prepare();

    let ui = MainWindow::new()?;
    ui.set_interp_available(interpolate::available());
    ui.set_tasks(ModelRc::new(VecModel::<TaskRow>::from(Vec::new())));
    ui.set_log_lines(ModelRc::new(VecModel::<slint::SharedString>::from(
        Vec::new(),
    )));

    let logger = ui.as_weak();
    match cuda_dir.as_deref() {
        Some(dir) => log(&logger, format!("CUDA runtime: {}", dir.display())),
        None => log(
            &logger,
            "GPU runtime missing — preview and export are paused to avoid unusably slow CPU processing",
        ),
    }

    let probe = gpu::probe();
    let auto_tile_size = recommended_tile();
    ui.set_gpu_name(probe.name.clone().into());

    // Readiness is decided by the CUDA payload, not by nvidia-smi. The driver
    // can happily report a device while ONNX Runtime still has no provider it
    // can load, and reporting RUNTIME OK there gives a false green light that
    // only fails once a session is built.
    ui.set_runtime_state(
        match (cuda_dir.is_some(), probe.device_present) {
            (true, true) => "RUNTIME OK",
            (false, true) => "GPU RUNTIME MISSING",
            (_, false) => "NO CUDA DEVICE",
        }
        .into(),
    );
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

    let all_models = upscale::registry();
    if all_models.is_empty() {
        ui.set_runtime_state("NO MODELS FOUND".into());
        log(&logger, "no models found — drop .onnx files into models/");
    } else {
        ui.set_models_count(all_models.len() as i32);
        log(
            &logger,
            format!(
                "{} models: {}",
                all_models.len(),
                all_models
                    .iter()
                    .map(|m| m.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
    }

    let video_models = Rc::new(models_for_mode(&all_models, 0));
    let image_models = Rc::new(models_for_mode(&all_models, 1));

    // Restore the previous session. The model is stored by name, so it
    // survives files being added to or removed from the models folder;
    // alphabetical order would otherwise put AnimeSharp first, the worst
    // possible default for photographic footage.
    let saved = settings::load();
    let initial_mode = saved.mode_index.clamp(0, 1);
    ui.set_mode_index(initial_mode);
    let initial_models = if initial_mode == 0 {
        video_models.as_ref().clone()
    } else {
        image_models.as_ref().clone()
    };
    let initial_name = if initial_mode == 0 {
        if saved.video_model.is_empty() {
            saved.model.as_str()
        } else {
            saved.video_model.as_str()
        }
    } else if saved.image_model.is_empty() {
        saved.model.as_str()
    } else {
        saved.image_model.as_str()
    };
    set_model_picker(&ui, &initial_models, initial_name);
    let active_models = Rc::new(RefCell::new(initial_models));
    let remembered_models = Rc::new(RefCell::new([
        saved.video_model.clone(),
        saved.image_model.clone(),
    ]));
    let current_mode = Rc::new(Cell::new(initial_mode));
    ui.set_scale_index(saved.scale_index);
    ui.set_strength(saved.strength);
    ui.set_detail_limit(saved.detail_limit);
    ui.set_fp16(saved.fp16);
    ui.set_auto_tile(saved.auto_tile);
    ui.set_content_index(saved.content_index);
    ui.set_restore_index(saved.restore_index);
    ui.set_detect_fields(saved.detect_fields);
    ui.set_preserve_grain(saved.preserve_grain);
    ui.set_match_color(saved.match_color);
    ui.set_stability_on(saved.stability);
    ui.set_lock(saved.lock);
    ui.set_cut_detect(saved.cut_detect);
    ui.set_interp_on(saved.interp_enabled && interpolate::available());
    ui.set_interp_index(saved.interp_index);
    ui.set_dock_panel(saved.dock_panel);
    ui.set_view_mode(saved.view_mode);
    log(&logger, format!("settings: {}", settings::path().display()));
    log(
        &logger,
        format!("persistent activity log: {}", activity_log_path().display()),
    );
    if interpolate::available() {
        log(
            &logger,
            format!(
                "RIFE interpolation model: {}",
                interpolate::model_path().display()
            ),
        );
    } else {
        log(
            &logger,
            format!(
                "RIFE interpolation unavailable: {} is missing",
                interpolate::model_path().display()
            ),
        );
    }

    let clip: Rc<RefCell<Option<media::ClipInfo>>> = Rc::new(RefCell::new(None));
    let still: Rc<RefCell<Option<image_mode::ImageInfo>>> = Rc::new(RefCell::new(None));
    let preview_generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tx = spawn_preview_worker(ui.as_weak(), preview_generation.clone());
    let image_tx = spawn_image_preview_worker(ui.as_weak(), preview_generation.clone());

    // Source frames decode off the event loop too, or scrubbing freezes the UI.
    let before_generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let before_tx = spawn_before_worker(ui.as_weak(), before_generation.clone());

    // Kick a preview render from whatever the controls currently say.
    let request = {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let still = still.clone();
        let active_models = active_models.clone();
        let tx = tx.clone();
        let image_tx = image_tx.clone();
        let preview_generation = preview_generation.clone();
        move || {
            let ui = handle.unwrap();
            let selected_model = active_models
                .borrow()
                .get(ui.get_model_index().max(0) as usize)
                .cloned();
            let Some(model) = selected_model else {
                return;
            };
            ui.set_model_note(upscale::guidance(&model.name).into());
            ui.set_model_is_video(model.is_video());
            ui.set_model_video_tuned(model.video_tuned);
            let target_scale = SCALES[(ui.get_scale_index().max(0) as usize).min(2)];
            ui.set_scale_note(
                upscale::scale_advice(&model, target_scale, &active_models.borrow())
                    .unwrap_or_default()
                    .into(),
            );

            if ui.get_mode_index() == 1 {
                let Some(info) = still.borrow().clone() else {
                    return;
                };
                ui.set_out_badge(
                    format!(
                        "{} × {} · {}",
                        info.width * target_scale,
                        info.height * target_scale,
                        model.name
                    )
                    .into(),
                );
                ui.set_preview_status("upscaling image preview…".into());
                let generation = preview_generation.fetch_add(1, Ordering::Relaxed) + 1;
                if ui.get_export_running() {
                    ui.set_preview_status(
                        "image loaded · preview deferred while export is running".into(),
                    );
                    return;
                }
                let _ = image_tx.send(ImageJob {
                    generation,
                    image: info,
                    model,
                    target_scale,
                    strength: ui.get_strength(),
                    detail_limit: ui.get_detail_limit(),
                    match_color: ui.get_match_color(),
                    fp16: ui.get_fp16(),
                    tile: if ui.get_auto_tile() {
                        auto_tile_size
                    } else {
                        SAFE_TILE
                    },
                });
                return;
            }

            let Some(info) = clip.borrow().clone() else {
                return;
            };
            let interpolation_fps = ui
                .get_interp_on()
                .then(|| interpolate::target_fps(ui.get_interp_index()));

            if let Some(fps) = interpolation_fps {
                if fps as f64 <= info.fps {
                    ui.set_preview_status(
                        format!("choose an interpolation target above {:.2} fps", info.fps).into(),
                    );
                    return;
                }
                if !interpolate::available() {
                    ui.set_preview_status("RIFE model missing — interpolation unavailable".into());
                    return;
                }
            }

            let at_secs = ui.get_playhead().clamp(0.0, 1.0) as f64 * info.duration;
            ui.set_out_badge(
                format!(
                    "{} × {} · {}{}",
                    info.width * target_scale,
                    info.height * target_scale,
                    model.name,
                    interpolation_fps
                        .map(|fps| format!(" · {fps} fps RIFE"))
                        .unwrap_or_default()
                )
                .into(),
            );
            ui.set_preview_status("upscaling sample frame…".into());

            // Replacing the clip or changing its settings invalidates an older
            // preview even when this request must wait for an active export.
            let generation = preview_generation.fetch_add(1, Ordering::Relaxed) + 1;

            if ui.get_export_running() {
                ui.set_preview_status(
                    "video loaded · preview deferred while export is running".into(),
                );
                log(
                    &ui.as_weak(),
                    "preview deferred: active export keeps exclusive use of the GPU",
                );
                return;
            }

            let _ = tx.send(Job {
                generation,
                clip: info,
                at_secs,
                model: model.clone(),
                target_scale,
                strength: ui.get_strength(),
                detail_limit: ui.get_detail_limit(),
                source_treatment: media::SourceTreatment {
                    content: ui.get_content_index(),
                    restore: ui.get_restore_index(),
                    deinterlace: ui.get_detect_fields(),
                },
                match_color: ui.get_match_color(),
                interpolation_fps,
                fp16: ui.get_fp16(),
                tile: if ui.get_auto_tile() {
                    auto_tile_size
                } else {
                    SAFE_TILE
                },
                stability: ui.get_stability_on() && !model.is_video(),
                lock_strength: ui.get_lock(),
                cut_detect: ui.get_cut_detect(),
            });
        }
    };

    let open = {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let still = still.clone();
        let request = request.clone();
        let before_tx = before_tx.clone();
        let before_generation = before_generation.clone();
        move |path: std::path::PathBuf| {
            let Some(ui) = handle.upgrade() else { return };
            let image_path = image_mode::is_supported(&path);
            if image_path && ui.get_mode_index() != 1 {
                ui.set_mode_index(1);
                ui.invoke_mode_changed(1);
            } else if !image_path && ui.get_mode_index() != 0 {
                ui.set_mode_index(0);
                ui.invoke_mode_changed(0);
            }
            if ui.get_mode_index() == 1 {
                match image_mode::probe(&path) {
                    Ok(info) => {
                        log(
                            &ui.as_weak(),
                            format!(
                                "opened image {} · {}×{} · {} / {}",
                                info.file_name(),
                                info.width,
                                info.height,
                                info.codec,
                                info.pix_fmt
                            ),
                        );
                        load_image(&ui, &info);
                        *still.borrow_mut() = Some(info);
                        request();
                    }
                    Err(error) => {
                        log(
                            &ui.as_weak(),
                            format!("could not read image {}: {error}", path.display()),
                        );
                        ui.set_preview_status(format!("could not read image: {error}").into());
                    }
                }
                return;
            }
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
                    let generation = before_generation.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = before_tx.send(BeforeJob {
                        generation,
                        clip: info.clone(),
                        at_secs: 0.0,
                        treatment: media::SourceTreatment {
                            content: ui.get_content_index(),
                            restore: ui.get_restore_index(),
                            deinterlace: ui.get_detect_fields(),
                        },
                        thumbnail: true,
                    });
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
        let handle = ui.as_weak();
        ui.on_import_clicked(move || {
            let image_mode = handle
                .upgrade()
                .map(|ui| ui.get_mode_index() == 1)
                .unwrap_or(false);
            let mut dialog = rfd::FileDialog::new();
            dialog = if image_mode {
                dialog.add_filter(
                    "Images",
                    &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                )
            } else {
                dialog.add_filter("Video", &["mp4", "mkv", "mov", "avi", "webm", "m2ts", "ts"])
            };
            if let Some(path) = dialog.add_filter("All files", &["*"]).pick_file() {
                open(path);
            }
        });
    }

    {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let request = request.clone();
        let before_tx = before_tx.clone();
        let before_generation = before_generation.clone();
        ui.on_seek(move |pos| {
            let ui = handle.unwrap();
            if ui.get_mode_index() == 1 {
                return;
            }
            if let Some(info) = clip.borrow().as_ref() {
                // Queued, not decoded here: this runs on the event loop and a
                // blocking ffmpeg call would freeze the window every scrub step.
                let generation = before_generation.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = before_tx.send(BeforeJob {
                    generation,
                    clip: info.clone(),
                    at_secs: pos.clamp(0.0, 1.0) as f64 * info.duration,
                    treatment: media::SourceTreatment {
                        content: ui.get_content_index(),
                        restore: ui.get_restore_index(),
                        deinterlace: ui.get_detect_fields(),
                    },
                    thumbnail: false,
                });
            }
            request();
        });
    }

    {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let still = still.clone();
        let active_models = active_models.clone();
        let video_models = video_models.clone();
        let image_models = image_models.clone();
        let remembered_models = remembered_models.clone();
        let current_mode = current_mode.clone();
        let request = request.clone();
        let before_tx = before_tx.clone();
        let before_generation = before_generation.clone();
        let preview_generation = preview_generation.clone();
        ui.on_mode_changed(move |requested_mode| {
            let Some(ui) = handle.upgrade() else { return };
            let next_mode = requested_mode.clamp(0, 1);
            let previous_mode = current_mode.replace(next_mode);
            if let Some(model) = active_models
                .borrow()
                .get(ui.get_model_index().max(0) as usize)
            {
                remembered_models.borrow_mut()[previous_mode as usize] = model.name.clone();
            }

            let next_models = if next_mode == 0 {
                video_models.as_ref().clone()
            } else {
                image_models.as_ref().clone()
            };
            let selected = remembered_models.borrow()[next_mode as usize].clone();
            *active_models.borrow_mut() = next_models;
            set_model_picker(&ui, &active_models.borrow(), &selected);

            preview_generation.fetch_add(1, Ordering::Relaxed);
            clear_source_display(&ui);
            if next_mode == 0 {
                if let Some(info) = clip.borrow().clone() {
                    load_clip(&ui, &info);
                    let generation = before_generation.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = before_tx.send(BeforeJob {
                        generation,
                        clip: info,
                        at_secs: 0.0,
                        treatment: media::SourceTreatment {
                            content: ui.get_content_index(),
                            restore: ui.get_restore_index(),
                            deinterlace: ui.get_detect_fields(),
                        },
                        thumbnail: true,
                    });
                    request();
                }
            } else if let Some(info) = still.borrow().clone() {
                load_image(&ui, &info);
                request();
            }

            settings::save(&collect_settings(
                &ui,
                &active_models.borrow(),
                &remembered_models.borrow(),
            ));
            log(
                &ui.as_weak(),
                if next_mode == 0 {
                    "switched to Video mode"
                } else {
                    "switched to Image mode"
                },
            );
        });
    }

    // Every control funnels through here, so persistence hangs off the same
    // signal that triggers a re-render.
    {
        let handle = ui.as_weak();
        let active_models = active_models.clone();
        let remembered_models = remembered_models.clone();
        let request = request.clone();
        ui.on_settings_changed(move || {
            if let Some(ui) = handle.upgrade() {
                settings::save(&collect_settings(
                    &ui,
                    &active_models.borrow(),
                    &remembered_models.borrow(),
                ));
            }
            request();
        });
    }

    {
        let request = request.clone();
        ui.on_resume_preview(request);
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
    let queue: Rc<RefCell<Vec<TaskSpec>>> = Rc::new(RefCell::new(Vec::new()));
    let active_cancel: Arc<Mutex<Option<Arc<AtomicBool>>>> = Arc::new(Mutex::new(None));
    let export_tx = spawn_exporter(ui.as_weak(), active_cancel.clone());

    {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let still = still.clone();
        let preview_generation = preview_generation.clone();
        ui.on_clear_video(move || {
            let Some(ui) = handle.upgrade() else { return };
            preview_generation.fetch_add(1, Ordering::Relaxed);
            let image_mode = ui.get_mode_index() == 1;
            if image_mode {
                still.borrow_mut().take();
            } else {
                clip.borrow_mut().take();
            }
            clear_source_display(&ui);
            ui.set_preview_status(
                if image_mode {
                    "image cleared"
                } else {
                    "video cleared"
                }
                .into(),
            );
            log(
                &ui.as_weak(),
                if image_mode {
                    "current image cleared"
                } else {
                    "current video cleared"
                },
            );
        });
    }

    {
        let handle = ui.as_weak();
        let queue = queue.clone();
        let active_cancel = active_cancel.clone();
        ui.on_clear_queue(move || {
            if let Ok(slot) = active_cancel.lock() {
                if let Some(cancel) = slot.as_ref() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            queue.borrow_mut().clear();
            if let Some(ui) = handle.upgrade() {
                if let Some(vm) = ui.get_tasks().as_any().downcast_ref::<VecModel<TaskRow>>() {
                    vm.clear();
                }
                log(&ui.as_weak(), "export queue cleared");
            }
        });
    }

    {
        let handle = ui.as_weak();
        let clip = clip.clone();
        let still = still.clone();
        let active_models = active_models.clone();
        let queue = queue.clone();
        ui.on_add_task(move || {
            let ui = handle.unwrap();
            let Some(model) = active_models
                .borrow()
                .get(ui.get_model_index().max(0) as usize)
                .cloned()
            else {
                return;
            };
            let target_scale = SCALES[(ui.get_scale_index().max(0) as usize).min(2)];
            let strength = ui.get_strength();
            let detail_limit = ui.get_detail_limit();

            if ui.get_mode_index() == 1 {
                let Some(info) = still.borrow().clone() else {
                    return;
                };
                let stem = info
                    .path
                    .file_stem()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "image".into());
                let candidate = info
                    .path
                    .with_file_name(format!("{stem}_{target_scale}x_{}.png", model.name));
                let out_path = unique_output_path(&candidate, &queue.borrow());
                let spec = image_mode::ExportSpec {
                    image: info.clone(),
                    model: model.clone(),
                    target_scale,
                    strength,
                    detail_limit,
                    match_color: ui.get_match_color(),
                    fp16: ui.get_fp16(),
                    tile: if ui.get_auto_tile() {
                        auto_tile_size
                    } else {
                        SAFE_TILE
                    },
                    out_path: out_path.clone(),
                };
                let (width, height) = spec.output_size();
                let row = TaskRow {
                    name: info.file_name().into(),
                    detail: format!(
                        "{width}×{height} · {} · lossless PNG · str {strength:.2} · lim {detail_limit:.2}",
                        model.name
                    )
                    .into(),
                    status: "queued".into(),
                    path: out_path.display().to_string().into(),
                    progress: 0.0,
                    running: false,
                    failed: false,
                    done: false,
                };
                queue.borrow_mut().push(TaskSpec::Image(spec));
                if let Some(rows) = ui.get_tasks().as_any().downcast_ref::<VecModel<TaskRow>>() {
                    rows.push(row);
                }
                return;
            }

            let Some(info) = clip.borrow().clone() else {
                return;
            };
            let interpolation_fps = ui
                .get_interp_on()
                .then(|| interpolate::target_fps(ui.get_interp_index()));
            if let Some(fps) = interpolation_fps {
                if fps as f64 <= info.fps {
                    ui.set_preview_status(
                        format!("choose an interpolation target above {:.2} fps", info.fps).into(),
                    );
                    return;
                }
                if !interpolate::available() {
                    ui.set_preview_status("RIFE model missing — task not added".into());
                    log(&ui.as_weak(), "task rejected: RIFE interpolation model is missing");
                    return;
                }
            }

            let stem = info
                .path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".into());
            let fps_suffix = interpolation_fps
                .map(|fps| format!("_{fps}fps"))
                .unwrap_or_default();
            let candidate = info.path.with_file_name(format!(
                "{stem}_{target_scale}x_{}{fps_suffix}.mp4",
                model.name
            ));
            let out_path = unique_output_path(&candidate, &queue.borrow());

            let spec = export::ExportSpec {
                clip: info.clone(),
                model: model.clone(),
                target_scale,
                strength,
                detail_limit,
                source_treatment: media::SourceTreatment {
                    content: ui.get_content_index(),
                    restore: ui.get_restore_index(),
                    deinterlace: ui.get_detect_fields(),
                },
                match_color: ui.get_match_color(),
                stability: ui.get_stability_on() && !model.is_video(),
                lock_strength: ui.get_lock(),
                cut_detect: ui.get_cut_detect(),
                interpolation_fps,
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
                    "{ow}×{oh} · {} · {} fps · str {strength:.2} · lim {detail_limit:.2} · {} frames",
                    model.name,
                    spec.output_fps(),
                    spec.output_frames()
                )
                .into(),
                status: "queued".into(),
                path: out_path.display().to_string().into(),
                progress: 0.0,
                running: false,
                failed: false,
                done: false,
            };

            queue.borrow_mut().push(TaskSpec::Video(spec));
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
        let active_models = active_models.clone();
        let remembered_models = remembered_models.clone();
        let active_cancel = active_cancel.clone();
        ui.window().on_close_requested(move || {
            if let Ok(slot) = active_cancel.lock() {
                if let Some(cancel) = slot.as_ref() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            if let Some(ui) = handle.upgrade() {
                settings::save(&collect_settings(
                    &ui,
                    &active_models.borrow(),
                    &remembered_models.borrow(),
                ));
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
                let hover_ui = ui.as_weak();
                match dnd::attach(ui.window(), open, move |active| {
                    if let Some(ui) = hover_ui.upgrade() {
                        ui.set_drop_active(active);
                    }
                }) {
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
fn spawn_preview_worker(
    handle: slint::Weak<MainWindow>,
    current_generation: Arc<std::sync::atomic::AtomicU64>,
) -> Sender<Job> {
    let (tx, rx) = channel::<Job>();

    std::thread::spawn(move || {
        let mut loaded: Option<(String, bool, upscale::Engine)> = None;
        let mut rife_model: Option<interpolate::Rife> = None;

        while let Ok(job) = rx.recv() {
            // Drain to the newest request — scrubbing and dragging sliders
            // queues jobs faster than they can be served, and only the last
            // one is worth rendering.
            let job = std::iter::successors(Some(job), |_| rx.try_recv().ok())
                .last()
                .unwrap();

            if current_generation.load(Ordering::Relaxed) != job.generation {
                continue;
            }

            let started = std::time::Instant::now();

            let (source, interpolation_note) =
                match decode_preview_source(&job, &mut rife_model, &handle) {
                    Ok(result) => result,
                    Err(e) => {
                        log(&handle, format!("preview source processing failed: {e}"));
                        report_current(
                            &handle,
                            &current_generation,
                            job.generation,
                            "source processing failed — full details are in the activity log"
                                .into(),
                            None,
                        );
                        continue;
                    }
                };

            if current_generation.load(Ordering::Relaxed) == job.generation {
                let h = handle.clone();
                let source_width = source.width;
                let source_height = source.height;
                let source_rgb = source.rgb.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = h.upgrade() {
                        ui.set_frame_before(to_image(&media::Frame {
                            width: source_width,
                            height: source_height,
                            rgb: source_rgb,
                        }));
                    }
                });
            }

            let listed_scale = job.model.scale.max(1);
            let preview_tile = job.tile.min(4096 / listed_scale).max(64);
            let step = preview_tile.saturating_sub(OVERLAP * 2).max(16);
            let tiles = source.width.div_ceil(step) * source.height.div_ceil(step);
            log(
                &handle,
                format!(
                    "preview started: {} at {:.2}s · {}×{} → {}×{} · {} · {}px tiles ({tiles} total)",
                    job.clip.file_name(),
                    job.at_secs,
                    source.width,
                    source.height,
                    source.width.saturating_mul(job.target_scale),
                    source.height.saturating_mul(job.target_scale),
                    job.model.name,
                    preview_tile,
                ),
            );

            let stale = loaded
                .as_ref()
                .map(|(n, f, _)| n != &job.model.name || *f != job.fp16)
                .unwrap_or(true);
            if stale {
                log(
                    &handle,
                    format!(
                        "preview: loading {} ({})",
                        job.model.name,
                        if job.fp16 { "fp16 preferred" } else { "fp32" }
                    ),
                );
                match upscale::Engine::load(&job.model, job.fp16) {
                    Ok(u) => {
                        log(
                            &handle,
                            format!(
                                "loaded {} · {}x native · {} · {}",
                                job.model.name,
                                u.scale(),
                                if job.fp16 { "fp16" } else { "fp32" },
                                u.provider()
                            ),
                        );
                        // Report what ORT actually registered, not what was
                        // requested — CUDA can silently fall back.
                        let h = handle.clone();
                        let p = u.provider().to_string();
                        let _ = slint::invoke_from_event_loop(move || {
                            if let Some(ui) = h.upgrade() {
                                ui.set_runtime_state(
                                    if p == "CUDA" {
                                        "RUNTIME OK"
                                    } else {
                                        "CPU FALLBACK"
                                    }
                                    .into(),
                                );
                                ui.set_rt_provider(p.into());
                            }
                        });
                        loaded = Some((job.model.name.clone(), job.fp16, u));
                    }
                    Err(e) => {
                        log(&handle, format!("model load failed: {e}"));
                        report_current(
                            &handle,
                            &current_generation,
                            job.generation,
                            format!("model load failed: {e}"),
                            None,
                        );
                        continue;
                    }
                }
            }

            let Some((_, _, up)) = loaded.as_mut() else {
                continue;
            };

            if up.provider() != "CUDA" {
                let message = "GPU runtime unavailable — preview stopped instead of running an impractically slow CPU render";
                log(&handle, message);
                report_current(
                    &handle,
                    &current_generation,
                    job.generation,
                    message.into(),
                    None,
                );
                continue;
            }
            let preview_tile = up.effective_tile(preview_tile);

            // Feed tile progress back so the status bar fills as the frame
            // resolves rather than sitting blank for several seconds.
            let progress_handle = handle.clone();
            let progress_started = std::time::Instant::now();
            let progress_generation = current_generation.clone();
            let job_generation = job.generation;
            let mut on_progress = move |done: u32, total: u32| {
                if progress_generation.load(Ordering::Relaxed) != job_generation {
                    return;
                }
                let frac = if total == 0 {
                    0.0
                } else {
                    done as f32 / total as f32
                };
                if done > 0 {
                    let elapsed = progress_started.elapsed().as_secs_f64();
                    let finish = if done < total {
                        format!(
                            " · {} remaining",
                            human_duration(elapsed / done as f64 * (total - done) as f64)
                        )
                    } else {
                        " · done".into()
                    };
                    log(
                        &progress_handle,
                        format!(
                            "preview: tile {done}/{total} ({:.0}%) · {} elapsed{finish}",
                            frac * 100.0,
                            human_duration(elapsed)
                        ),
                    );
                }
                let h = progress_handle.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = h.upgrade() {
                        ui.set_preview_progress(frac);
                    }
                });
            };

            // Post-processing chain, shared by the current frame and — when the
            // lock is being previewed — the preceding one.
            let finish = |out: media::Frame, src: &media::Frame| -> media::Frame {
                // Order matters: clamp structural error first, then mix toward
                // the source, then resample to the requested scale. Resampling
                // last means a 4x model asked for 2x gets its invented
                // high-frequency detail averaged away.
                let colored = if job.match_color {
                    upscale::match_color_to_source(&out, src)
                } else {
                    out
                };
                let limited = upscale::clamp_deviation(&colored, src, job.detail_limit * 255.0);
                let blended = upscale::blend_toward_source(&limited, src, job.strength);
                upscale::resample(
                    &blended,
                    src.width * job.target_scale,
                    src.height * job.target_scale,
                )
            };

            let preview_sources = if up.window() == 1 {
                vec![source.clone()]
            } else {
                let center = up.window() / 2;
                let step = 1.0
                    / job
                        .interpolation_fps
                        .map(|fps| fps as f64)
                        .unwrap_or(job.clip.fps)
                        .max(1.0);
                let mut frames = Vec::with_capacity(up.window());
                let mut failed = None;
                for index in 0..up.window() {
                    let mut neighbor = job.clone();
                    neighbor.at_secs = (job.at_secs
                        + (index as isize - center as isize) as f64 * step)
                        .clamp(0.0, job.clip.duration.max(job.at_secs));
                    match decode_preview_source(&neighbor, &mut rife_model, &handle) {
                        Ok((frame, _)) => frames.push(frame),
                        Err(e) => {
                            failed = Some(e);
                            break;
                        }
                    }
                }
                if let Some(e) = failed {
                    log(&handle, format!("video preview window failed: {e}"));
                    report_current(
                        &handle,
                        &current_generation,
                        job.generation,
                        format!("video preview failed: {e}"),
                        None,
                    );
                    continue;
                }
                frames
            };
            let preview_pick = preview_sources.len() / 2;

            match up.run_batch(&preview_sources, preview_tile, OVERLAP, &mut on_progress) {
                Ok(outputs) => {
                    let Some(out) = outputs.into_iter().nth(preview_pick) else {
                        report_current(
                            &handle,
                            &current_generation,
                            job.generation,
                            "video preview returned no center frame".into(),
                            None,
                        );
                        continue;
                    };
                    let mut final_frame = finish(out, &source);
                    let mut lock_note = None;

                    // Stability needs two frames to mean anything, so the
                    // preview renders the preceding one as well. Without this
                    // the Lock and Cut detect sliders can only be judged by
                    // running a full export.
                    if job.stability && up.window() == 1 && job.at_secs > 0.0 {
                        let step = 1.0 / job.clip.fps.max(1.0);
                        let prev_at = (job.at_secs - step).max(0.0);
                        match media::frame_at_with(&job.clip, prev_at, None, job.source_treatment)
                            .and_then(|prev_source| {
                                let prev_out = up
                                    .run_batch(
                                        std::slice::from_ref(&prev_source),
                                        preview_tile,
                                        OVERLAP,
                                        &mut |_, _| {},
                                    )?
                                    .into_iter()
                                    .next()
                                    .ok_or_else(|| {
                                        "previous preview returned no frame".to_string()
                                    })?;
                                let prev_final = finish(prev_out, &prev_source);
                                let motion =
                                    temporal::analyze_with(&prev_source, &source, job.cut_detect)?;
                                let scale = final_frame.width as f32 / source.width.max(1) as f32;
                                Ok((
                                    temporal::lock(
                                        &prev_final,
                                        &final_frame,
                                        &motion,
                                        job.lock_strength,
                                        scale,
                                    ),
                                    motion.mean_confidence(),
                                    motion.cut,
                                ))
                            }) {
                            Ok((locked, confidence, cut)) => {
                                final_frame = locked;
                                lock_note = Some(if cut {
                                    "lock released at cut".to_string()
                                } else {
                                    format!("lock {:.2} × held {confidence:.2}", job.lock_strength)
                                });
                            }
                            Err(e) => {
                                log(&handle, format!("preview lock skipped: {e}"));
                            }
                        }
                    }

                    let secs = started.elapsed().as_secs_f32();
                    let mut notes = Vec::new();
                    if let Some(note) = interpolation_note {
                        notes.push(note);
                    }
                    if let Some(note) = lock_note {
                        notes.push(note);
                    }
                    if job.match_color {
                        notes.push("color matched".into());
                    }
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
                        up.scale(),
                        job.target_scale,
                        extra,
                        if job.fp16 { "fp16" } else { "fp32" },
                        secs,
                        up.provider()
                    );
                    log(&handle, format!("preview completed: {note}"));
                    report_current(
                        &handle,
                        &current_generation,
                        job.generation,
                        note,
                        Some(final_frame),
                    );
                }
                Err(e) => {
                    log(&handle, format!("preview failed: {e}"));
                    report_current(
                        &handle,
                        &current_generation,
                        job.generation,
                        format!("inference failed: {e}"),
                        None,
                    );
                }
            }
        }
    });

    tx
}

fn decode_preview_source(
    job: &Job,
    rife_model: &mut Option<interpolate::Rife>,
    handle: &slint::Weak<MainWindow>,
) -> Result<(media::Frame, Option<String>), String> {
    let Some(target_fps) = job
        .interpolation_fps
        .filter(|fps| *fps as f64 > job.clip.fps)
    else {
        return media::frame_at_with(&job.clip, job.at_secs, None, job.source_treatment)
            .map(|frame| (frame, None));
    };

    if job.clip.frames < 2 {
        return media::frame_at_with(&job.clip, job.at_secs, None, job.source_treatment)
            .map(|frame| (frame, Some(format!("RIFE {target_fps} fps · single frame"))));
    }

    let position = job.at_secs.max(0.0) * job.clip.fps.max(1.0);
    let pair = (position.floor() as u64).min(job.clip.frames.saturating_sub(2));
    let exact_phase = (position - pair as f64).clamp(0.0, 1.0) as f32;
    // Timeline positions commonly land exactly on a source frame. Showing the
    // midpoint there makes enabling RIFE produce a real, visible preview.
    let phase = if !(0.001..=0.999).contains(&exact_phase) {
        0.5
    } else {
        exact_phase
    };
    let left = media::frame_at_with(
        &job.clip,
        pair as f64 / job.clip.fps,
        None,
        job.source_treatment,
    )?;
    let right = media::frame_at_with(
        &job.clip,
        (pair + 1) as f64 / job.clip.fps,
        None,
        job.source_treatment,
    )?;

    if rife_model.is_none() {
        log(handle, "preview: loading RIFE timestep model on CUDA");
        let rife = interpolate::Rife::load()?;
        log(
            handle,
            format!("loaded RIFE interpolation model on {}", rife.provider),
        );
        *rife_model = Some(rife);
    }
    let rife = rife_model
        .as_mut()
        .ok_or_else(|| "RIFE model did not remain loaded".to_string())?;
    let cut = interpolate::is_scene_cut(&left, &right);
    let mut generated = rife.run(&left, &right, phase)?;
    if job.match_color && !cut {
        let reference = interpolate::color_reference(&left, &right, phase);
        generated = upscale::match_color_to_source(&generated, &reference);
    }

    Ok((
        generated,
        Some(format!("RIFE {target_fps} fps · t={phase:.2}")),
    ))
}

/// Runs still-image preview inference without any timeline or temporal stages.
fn spawn_image_preview_worker(
    handle: slint::Weak<MainWindow>,
    current_generation: Arc<std::sync::atomic::AtomicU64>,
) -> Sender<ImageJob> {
    let (tx, rx) = channel::<ImageJob>();

    std::thread::spawn(move || {
        let mut loaded: Option<(String, bool, upscale::Engine)> = None;
        while let Ok(job) = rx.recv() {
            let job = std::iter::successors(Some(job), |_| rx.try_recv().ok())
                .last()
                .unwrap();
            if current_generation.load(Ordering::Relaxed) != job.generation {
                continue;
            }

            let started = std::time::Instant::now();
            let source = match image_mode::decode(&job.image) {
                Ok(frame) => frame,
                Err(error) => {
                    log(&handle, format!("image preview decode failed: {error}"));
                    report_current(
                        &handle,
                        &current_generation,
                        job.generation,
                        format!("image decode failed: {error}"),
                        None,
                    );
                    continue;
                }
            };

            if current_generation.load(Ordering::Relaxed) == job.generation {
                let h = handle.clone();
                let before = source.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = h.upgrade() {
                        let image = to_image(&before);
                        ui.set_frame_before(image.clone());
                        ui.set_thumb(image);
                    }
                });
            }

            let stale = loaded
                .as_ref()
                .map(|(name, half, _)| name != &job.model.name || *half != job.fp16)
                .unwrap_or(true);
            if stale {
                log(
                    &handle,
                    format!("image preview: loading {}", job.model.name),
                );
                match upscale::Engine::load(&job.model, job.fp16) {
                    Ok(engine) => {
                        if engine.window() != 1 {
                            report_current(
                                &handle,
                                &current_generation,
                                job.generation,
                                "video-only model cannot be used for images".into(),
                                None,
                            );
                            continue;
                        }
                        loaded = Some((job.model.name.clone(), job.fp16, engine));
                    }
                    Err(error) => {
                        log(&handle, format!("image model load failed: {error}"));
                        report_current(
                            &handle,
                            &current_generation,
                            job.generation,
                            format!("model load failed: {error}"),
                            None,
                        );
                        continue;
                    }
                }
            }

            let Some((_, _, engine)) = loaded.as_mut() else {
                continue;
            };
            if engine.provider() != "CUDA" {
                report_current(
                    &handle,
                    &current_generation,
                    job.generation,
                    "GPU runtime unavailable — image preview stopped".into(),
                    None,
                );
                continue;
            }

            let progress_handle = handle.clone();
            let progress_generation = current_generation.clone();
            let wanted = job.generation;
            let mut on_progress = move |done: u32, total: u32| {
                if progress_generation.load(Ordering::Relaxed) != wanted {
                    return;
                }
                let fraction = if total == 0 {
                    0.0
                } else {
                    done as f32 / total as f32
                };
                let h = progress_handle.clone();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = h.upgrade() {
                        ui.set_preview_progress(fraction);
                    }
                });
            };

            let output = engine.run_batch(
                std::slice::from_ref(&source),
                job.tile,
                OVERLAP,
                &mut on_progress,
            );
            match output.and_then(|frames| {
                frames
                    .into_iter()
                    .next()
                    .ok_or_else(|| "image model returned no frame".to_string())
            }) {
                Ok(model_output) => {
                    let spec = image_mode::ExportSpec {
                        image: job.image.clone(),
                        model: job.model.clone(),
                        target_scale: job.target_scale,
                        strength: job.strength,
                        detail_limit: job.detail_limit,
                        match_color: job.match_color,
                        fp16: job.fp16,
                        tile: job.tile,
                        out_path: std::path::PathBuf::new(),
                    };
                    let final_image = image_mode::finish(&spec, model_output, &source);
                    let note = format!(
                        "{} · {}x native → {}x · lossless PNG · {:.1}s on {}",
                        job.model.name,
                        engine.scale(),
                        job.target_scale,
                        started.elapsed().as_secs_f32(),
                        engine.provider()
                    );
                    log(&handle, format!("image preview completed: {note}"));
                    report_current(
                        &handle,
                        &current_generation,
                        job.generation,
                        note,
                        Some(final_image),
                    );
                }
                Err(error) => {
                    log(&handle, format!("image preview failed: {error}"));
                    report_current(
                        &handle,
                        &current_generation,
                        job.generation,
                        format!("image preview failed: {error}"),
                        None,
                    );
                }
            }
        }
    });

    tx
}

/// Runs full-clip exports off the UI thread.
fn spawn_exporter(
    handle: slint::Weak<MainWindow>,
    active_cancel: Arc<Mutex<Option<Arc<AtomicBool>>>>,
) -> Sender<Vec<(usize, TaskSpec)>> {
    let (tx, rx) = channel::<Vec<(usize, TaskSpec)>>();

    std::thread::spawn(move || {
        let mut loaded: Option<(String, bool, upscale::Engine)> = None;
        let mut rife_model: Option<interpolate::Rife> = None;

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

                let model = spec.model().clone();
                let fp16 = spec.fp16();

                let stale = loaded
                    .as_ref()
                    .map(|(name, half, _)| name != &model.name || *half != fp16)
                    .unwrap_or(true);
                if stale {
                    match upscale::Engine::load(&model, fp16) {
                        Ok(u) => {
                            log(
                                &handle,
                                format!(
                                    "export: loaded {} ({}x native) on {}",
                                    model.name,
                                    u.scale(),
                                    u.provider()
                                ),
                            );
                            loaded = Some((model.name.clone(), fp16, u));
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
                if up.provider() != "CUDA" {
                    log(
                        &handle,
                        "export stopped: GPU runtime unavailable; slow CPU processing is disabled",
                    );
                    set_task(&handle, idx, 0.0, "GPU runtime missing", false, true);
                    continue;
                }

                let needs_rife = matches!(
                    &spec,
                    TaskSpec::Video(video) if video.interpolation_fps.is_some()
                );
                if needs_rife && rife_model.is_none() {
                    set_task(&handle, idx, 0.0, "loading RIFE", true, false);
                    match interpolate::Rife::load() {
                        Ok(rife) => {
                            log(
                                &handle,
                                format!("export: loaded RIFE interpolation on {}", rife.provider),
                            );
                            rife_model = Some(rife);
                        }
                        Err(e) => {
                            log(&handle, format!("export: RIFE load failed: {e}"));
                            set_task(&handle, idx, 0.0, "RIFE failed", false, true);
                            continue;
                        }
                    }
                }

                match &spec {
                    TaskSpec::Video(video) => log(
                        &handle,
                        format!(
                            "video export started: {} · {} frames at {} fps{} · color match {}",
                            video.clip.file_name(),
                            video.output_frames(),
                            video.output_fps(),
                            video
                                .interpolation_fps
                                .map(|_| " · RIFE")
                                .unwrap_or_default(),
                            if video.match_color { "on" } else { "off" }
                        ),
                    ),
                    TaskSpec::Image(image) => log(
                        &handle,
                        format!(
                            "image export started: {} · {}×{} → {}×{} · {} · color match {}",
                            image.image.file_name(),
                            image.image.width,
                            image.image.height,
                            image.output_size().0,
                            image.output_size().1,
                            image.model.name,
                            if image.match_color { "on" } else { "off" }
                        ),
                    ),
                }

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
                    // `total` is an estimate from probe metadata, while `done`
                    // counts frames the decoder actually produced. VFR sources
                    // and CFR conversion can push done past total, so once that
                    // happens stop quoting a total or an ETA rather than
                    // showing "134/120" and a bar stuck at 100%.
                    let label = if done > total {
                        format!("{done} frames")
                    } else if done >= 2 {
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

                let result = match &spec {
                    TaskSpec::Video(video) => export::run(
                        video,
                        up,
                        rife_model.as_mut(),
                        video.tile,
                        OVERLAP,
                        cancel.clone(),
                        &mut on_progress,
                    )
                    .map(Some),
                    TaskSpec::Image(image) => {
                        image_mode::run(image, up, OVERLAP, cancel.as_ref(), &mut on_progress)
                            .map(|_| None)
                    }
                };

                match result {
                    Ok(stages) => {
                        let wall = started.elapsed();
                        let duration = human_duration(wall.as_secs_f64());
                        set_task(
                            &handle,
                            idx,
                            1.0,
                            &format!("done · {duration}"),
                            false,
                            false,
                        );
                        log(
                            &handle,
                            format!("exported {} in {duration}", spec.out_path().display()),
                        );
                        // Phase 0 baseline: where the time actually went, so
                        // later pipeline stages have something to be measured
                        // against rather than a single opaque total.
                        if let Some(stages) = stages {
                            log(&handle, stages.summary(wall));
                        }
                    }
                    Err(e) if e == "cancelled" => {
                        log(
                            &handle,
                            format!("export cancelled: {}", spec.out_path().display()),
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

/// A source-frame decode request for the before-pane.
struct BeforeJob {
    generation: u64,
    clip: media::ClipInfo,
    at_secs: f64,
    treatment: media::SourceTreatment,
    /// Also populate the video-info thumbnail, which only the initial load
    /// needs.
    thumbnail: bool,
}

/// Decodes before-pane frames off the UI thread.
///
/// `media::frame_at` spawns ffmpeg and blocks until a full rgb24 frame comes
/// back. Doing that on the Slint event loop froze the whole window for the
/// duration of every scrub step, which is very visible on long GOPs or a slow
/// disk. The model preview already used a worker; this gives the source frame
/// the same treatment.
fn spawn_before_worker(
    handle: slint::Weak<MainWindow>,
    generation: Arc<std::sync::atomic::AtomicU64>,
) -> Sender<BeforeJob> {
    let (tx, rx) = channel::<BeforeJob>();

    std::thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            // Scrubbing queues faster than ffmpeg can serve; only the newest
            // position is worth decoding.
            let job = std::iter::successors(Some(job), |_| rx.try_recv().ok())
                .last()
                .unwrap();
            if job.generation < generation.load(Ordering::Relaxed) {
                continue;
            }

            let decoded = media::frame_at_with(&job.clip, job.at_secs, None, job.treatment);
            let Ok(frame) = decoded else { continue };

            let h = handle.clone();
            let wanted = job.generation;
            let gen = generation.clone();
            let thumbnail = job.thumbnail;
            let posted = slint::invoke_from_event_loop(move || {
                // Re-check on the UI thread: the clip may have changed while
                // this frame was in flight.
                if wanted < gen.load(Ordering::Relaxed) {
                    return;
                }
                let Some(ui) = h.upgrade() else { return };
                let image = to_image(&frame);
                ui.set_frame_before(image.clone());
                if thumbnail {
                    ui.set_frame_after(image.clone());
                    ui.set_thumb(image);
                }
            });
            if posted.is_err() {
                return;
            }
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

fn activity_log_path() -> std::path::PathBuf {
    settings::path().with_file_name("activity.log")
}

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

    let path = activity_log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{stamp}  {msg}");
    }

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

fn unique_output_path(candidate: &std::path::Path, queue: &[TaskSpec]) -> std::path::PathBuf {
    let reserved =
        |p: &std::path::Path| p.exists() || queue.iter().any(|item| item.out_path() == p);
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
            // Requests made during an export only bump the generation and
            // return, so without this the after-pane stays blank or stale
            // until the user happens to touch a control.
            if !running {
                ui.invoke_resume_preview();
            }
        }
    });
}

/// Snapshots the current control state for persistence.
fn collect_settings(
    ui: &MainWindow,
    models: &[upscale::ModelEntry],
    remembered: &[String; 2],
) -> settings::Settings {
    let current_model = models
        .get(ui.get_model_index().max(0) as usize)
        .map(|model| model.name.clone())
        .unwrap_or_default();
    let mut video_model = remembered[0].clone();
    let mut image_model = remembered[1].clone();
    if ui.get_mode_index() == 0 {
        video_model = current_model.clone();
    } else {
        image_model = current_model.clone();
    }
    settings::Settings {
        model: current_model,
        mode_index: ui.get_mode_index(),
        video_model,
        image_model,
        scale_index: ui.get_scale_index(),
        strength: ui.get_strength(),
        detail_limit: ui.get_detail_limit(),
        fp16: ui.get_fp16(),
        auto_tile: ui.get_auto_tile(),
        content_index: ui.get_content_index(),
        restore_index: ui.get_restore_index(),
        detect_fields: ui.get_detect_fields(),
        preserve_grain: ui.get_preserve_grain(),
        match_color: ui.get_match_color(),
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

fn report_current(
    handle: &slint::Weak<MainWindow>,
    current_generation: &std::sync::atomic::AtomicU64,
    generation: u64,
    status: String,
    frame: Option<media::Frame>,
) {
    if current_generation.load(Ordering::Relaxed) == generation {
        report(handle, status, frame);
    }
}

fn clear_source_display(ui: &MainWindow) {
    ui.set_has_clip(false);
    ui.set_clip_name("".into());
    ui.set_clip_specs("".into());
    ui.set_src_badge("—".into());
    ui.set_out_badge("—".into());
    ui.set_source_note("".into());
    ui.set_preview_status("".into());
    ui.set_preview_progress(0.0);
    ui.set_frame_before(Image::default());
    ui.set_frame_after(Image::default());
    ui.set_thumb(Image::default());
    ui.set_zoom(1.0);
    ui.set_pan_x(0.0);
    ui.set_pan_y(0.0);
}

fn load_image(ui: &MainWindow, info: &image_mode::ImageInfo) {
    ui.set_clip_name(info.file_name().into());
    ui.set_clip_specs(info.specs().into());
    ui.set_src_badge(format!("{} × {} · {}", info.width, info.height, info.codec).into());
    ui.set_clip_aspect(info.width as f32 / info.height.max(1) as f32);
    ui.set_clip_duration(0.0);
    ui.set_clip_fps(0.0);
    ui.set_playhead(0.0);
    ui.set_zoom(1.0);
    ui.set_pan_x(0.0);
    ui.set_pan_y(0.0);
    ui.set_has_clip(true);
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
            "Interlaced ({}) detected — choose a TV profile and enable Deinterlace",
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
    // The first frame is decoded on the before-worker by the caller, so opening
    // a clip does not hitch the window before its first paint.
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

    #[test]
    fn video_and_image_modes_have_disjoint_model_lists() {
        let all = upscale::registry();
        let video = models_for_mode(&all, 0);
        let image = models_for_mode(&all, 1);
        assert!(!video.is_empty(), "video model list");
        assert!(!image.is_empty(), "image model list");
        assert!(video
            .iter()
            .all(|model| model.is_video() || model.video_tuned));
        assert!(image
            .iter()
            .all(|model| !model.is_video() && !model.video_tuned));
        assert_eq!(video.len() + image.len(), all.len());
    }

    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to benchmark real GPU tile sizes"]
    fn benchmark_preview_tile_sizes() {
        let path = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(std::path::PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let _ = cuda::prepare();
        let clip = media::probe(&path).expect("probe clip");
        let frame = media::frame_at(&clip, 0.0, None).expect("decode frame");
        let entry = upscale::registry()
            .into_iter()
            .find(|model| model.name == "4x-UltraSharp")
            .expect("find UltraSharp");
        let mut up = upscale::Upscaler::load(&entry, true).expect("load model");
        assert_eq!(up.provider, "CUDA");

        for tile in [512, 640, 896, 1024] {
            let started = std::time::Instant::now();
            let out = up
                .run(&frame, tile, OVERLAP, &mut |_, _| {})
                .unwrap_or_else(|e| panic!("tile {tile}: {e}"));
            eprintln!(
                "tile {tile}: {:.3}s for {}x{} -> {}x{}",
                started.elapsed().as_secs_f32(),
                frame.width,
                frame.height,
                out.width,
                out.height
            );
            drop(out);
        }
    }
}
