//! Commands execute on Slint's event loop through the same controls as a person.
use crate::{mcp_transport, MainWindow};
use serde_json::{json, Value};
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::{
    collections::HashMap,
    path::PathBuf,
    time::{Duration, Instant},
};

const LEASE: Duration = Duration::from_secs(12);

fn validate<'a>(
    values: &'a Value,
    schema: &Value,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    let values = values.as_object().ok_or("settings must be an object")?;
    if values.is_empty() {
        return Err("Provide at least one setting".into());
    }
    for (name, value) in values {
        let rule = schema.get(name).ok_or_else(|| {
            format!("Unknown or unavailable control: {name}. Use describe_controls.")
        })?;
        let valid = match rule["type"].as_str().unwrap_or("") {
            "boolean" => value.is_boolean(),
            "string" => value
                .as_str()
                .is_some_and(|s| s.chars().count() <= 4000 && !s.contains('\0')),
            "integer" => value.as_i64().is_some() && in_range(value, rule),
            "number" => in_range(value, rule),
            _ => false,
        };
        if !valid {
            return Err(format!("Invalid value for {name}; expected {rule}"));
        }
    }
    Ok(values)
}

fn in_range(value: &Value, rule: &Value) -> bool {
    value.as_f64().is_some_and(|n| {
        n.is_finite()
            && rule["minimum"].as_f64().is_none_or(|min| n >= min)
            && rule["maximum"].as_f64().is_none_or(|max| n <= max)
    })
}

fn strings(model: ModelRc<SharedString>) -> Vec<String> {
    model.iter().map(|s| s.to_string()).collect()
}

fn busy(ui: &MainWindow) -> bool {
    ui.get_export_running() || ui.get_tasks().iter().any(|row| row.running)
}

fn snapshot(ui: &MainWindow, source: Option<PathBuf>) -> Value {
    let tasks: Vec<Value> = ui.get_tasks().iter().enumerate().map(|(index,row)| json!({
        "index": index, "name": row.name.to_string(), "detail": row.detail.to_string(),
        "status": row.status.to_string(), "path": row.path.to_string(), "progress": row.progress,
        "running": row.running, "done": row.done, "failed": row.failed,
    })).collect();
    json!({
        "application": "kraken-upscale", "version": env!("CARGO_PKG_VERSION"),
        "project_root": std::env::current_exe().ok().and_then(|p|p.parent().map(|p|p.to_path_buf())),
        "pid": std::process::id(), "mode": if ui.get_mode_index() == 0 {"video"} else {"image"},
        "has_media": ui.get_has_clip(), "source_path": source, "source_name": ui.get_clip_name().to_string(),
        "source_specs": ui.get_clip_specs().to_string(), "source_note": ui.get_source_note().to_string(),
        "duration": ui.get_clip_duration(), "fps": ui.get_clip_fps(),
        "settings": settings(ui), "view": view(ui), "panel": panel(ui),
        "playing": ui.get_playing(), "playhead": ui.get_playhead(),
        "preview_status": ui.get_preview_status().to_string(), "preview_progress": ui.get_preview_progress(),
        "export_running": busy(ui), "tasks": tasks,
        "models": strings(ui.get_model_names()), "generative_models": strings(ui.get_generative_model_names()),
        "runtime": {"status":ui.get_runtime_state().to_string(),"rtx_vsr_ready":ui.get_rtx_vsr_ready(),
            "generative_ready":ui.get_generative_runtime_ready(),"interpolation_ready":ui.get_interp_available()},
        "gpu": {"device":ui.get_sys_device().to_string(),"usage_percent":ui.get_sys_usage(),
            "vram_free_gb":ui.get_sys_vram_free(),"vram_used_gb":ui.get_sys_vram_used()},
        "mcp": {"connected":ui.get_mcp_connected(),"clients":ui.get_mcp_clients().to_string(),
            "status":ui.get_mcp_status().to_string()},
        "output_policy":"Unique output filenames beside the source, exactly as in the app queue."
    })
}

fn panel(ui: &MainWindow) -> &'static str {
    match ui.get_right_panel() {
        0 => "input",
        1 => "enhance",
        2 => "motion",
        3 => "output",
        _ => match ui.get_dock_panel() {
            1 => "logs",
            2 => "system",
            3 => "mcp",
            _ => "closed",
        },
    }
}

fn open_panel(ui: &MainWindow, name: &str) -> Result<(), String> {
    let (dock, settings) = match name {
        "closed" => (0, -1),
        "logs" => (1, -1),
        "system" => (2, -1),
        "mcp" => (3, -1),
        "models" => (4, -1),
        "input" => (0, 0),
        "enhance" => (0, 1),
        "motion" if ui.get_mode_index() == 0 => (0, 2),
        "output" => (0, 3),
        _ => return Err("Unknown panel, or Motion is unavailable in Images mode".into()),
    };
    ui.set_dock_panel(dock);
    ui.set_right_panel(settings);
    Ok(())
}

fn require_source(ui: &MainWindow) -> Result<(), String> {
    if ui.get_has_clip() {
        Ok(())
    } else {
        Err("Import media into the app first".into())
    }
}

fn dispatch(
    ui: &MainWindow,
    command: &str,
    args: &Value,
    import: &dyn Fn(PathBuf) -> Result<(), String>,
    source: &dyn Fn() -> Option<PathBuf>,
) -> Result<Value, String> {
    if busy(ui)
        && matches!(
            command,
            "import_media"
                | "set_mode"
                | "configure"
                | "add_to_batch"
                | "upscale"
                | "batch_upscale"
                | "clear_queue"
                | "clear_media"
                | "release_gpu_memory"
        )
    {
        return Err("An upscale is scheduled or running. Inspect get_state, wait, or explicitly cancel_upscale first.".into());
    }
    match command {
        "get_state" => {}
        "describe_controls" => {
            return Ok(
                json!({"settings": settings_schema(ui), "view": view_schema(ui),
            "models":strings(ui.get_model_names()),"generative_models":strings(ui.get_generative_model_names()),
            "video2x_presets":strings(ui.get_video2x_preset_names()),
            "panels":["logs","system","input","enhance","motion","output","mcp","models","closed"],
            "notes":["Indexes are zero-based; labels are in index order.",
                "Call set_mode before choosing its model or pipeline. Import switches mode to match the file.",
                "configure updates the visible controls and calls the same settings-changed action as the UI.",
                "Preserve grain is planned and disabled, so it is not exposed.",
                "RTX VSR supplies its own temporal history; model strength, detail, tiling and color settings are inactive in RTX mode.",
                "frame_index 1 crops video to 16:9 before upscaling (crop_position 0 top, 0.5 centre, 1 bottom); the output goal then applies to the cropped picture.",
                "Video2X (video_pipeline_index 2) runs the external Video2X runtime with video2x_preset_index; model, strength, detail, tiling, color match and Stability are inactive there. Source treatment, RIFE, output goal, codec, quality and audio still apply.",
                "Outputs use unique names beside the source. Upscale runs the first unfinished queue item; Batch Upscale runs all unfinished items."]}),
            )
        }
        "configure" => {
            let schema = settings_schema(ui);
            let values = validate(&args["settings"], &schema)?;
            if (values.get("video_pipeline_index") == Some(&json!(1))
                || values.get("image_pipeline_index") == Some(&json!(2)))
                && !ui.get_rtx_vsr_ready()
            {
                return Err("RTX VSR runtime is unavailable in this app".into());
            }
            if values.get("video_pipeline_index") == Some(&json!(2)) && !ui.get_video2x_ready() {
                return Err("Video2X runtime is unavailable in this app".into());
            }
            if values.get("image_pipeline_index") == Some(&json!(1))
                && !ui.get_generative_runtime_ready()
            {
                return Err("Generative runtime or model package is unavailable".into());
            }
            if values.get("interp_on") == Some(&json!(true)) && !ui.get_interp_available() {
                return Err("RIFE interpolation is unavailable".into());
            }
            if values.contains_key("model_index")
                && ui.get_model_names().iter().all(|n| n.starts_with("("))
            {
                return Err("No compatible model installed".into());
            }
            if values.contains_key("generative_model_index") && !ui.get_generative_runtime_ready() {
                return Err("No complete generative runtime/model available".into());
            }
            apply_settings(ui, values);
            let selected = if values.keys().any(|k| {
                matches!(
                    k.as_str(),
                    "resolution_index"
                        | "video_format_index"
                        | "video_quality_index"
                        | "audio_index"
                        | "image_format_index"
                )
            }) {
                "output"
            } else if values.keys().any(|k| {
                matches!(
                    k.as_str(),
                    "content_index" | "restore_index" | "frame_index" | "crop_position" | "detect_fields"
                )
            }) {
                "input"
            } else if ui.get_mode_index() == 0
                && values.keys().any(|k| {
                    matches!(
                        k.as_str(),
                        "stability_on"
                            | "lock"
                            | "cut_detect"
                            | "interp_on"
                            | "interp_index"
                            | "match_color"
                    )
                })
            {
                "motion"
            } else {
                "enhance"
            };
            open_panel(ui, selected)?;
            ui.set_src_open(true);
            ui.set_det_open(true);
            ui.set_stb_open(true);
            ui.set_fi_open(true);
            ui.set_col_open(true);
            ui.invoke_settings_changed();
        }
        "set_view" => {
            let schema = view_schema(ui);
            apply_view(ui, validate(&args["settings"], &schema)?);
        }
        "set_mode" => {
            let mode = match args["mode"].as_str() {
                Some("video") => 0,
                Some("image") => 1,
                _ => return Err("mode must be video or image".into()),
            };
            if mode != ui.get_mode_index() {
                ui.set_mode_index(mode);
                ui.invoke_mode_changed(mode);
            }
        }
        "open_panel" => open_panel(ui, args["panel"].as_str().ok_or("panel must be a string")?)?,
        "import_media" => {
            let path = PathBuf::from(
                args["path"]
                    .as_str()
                    .ok_or("path must be a local absolute filename")?,
            );
            if !path.is_absolute() || !path.is_file() {
                return Err("path must name an existing absolute local file".into());
            }
            import(path)?;
            open_panel(ui, "input")?;
        }
        "preview" => {
            require_source(ui)?;
            ui.invoke_resume_preview();
        }
        "playback" => {
            require_source(ui)?;
            if ui.get_mode_index() != 0 {
                return Err("Playback is only available in Video mode".into());
            }
            let playing = args["playing"].as_bool().ok_or("playing must be boolean")?;
            ui.set_playing(playing);
            ui.invoke_playback_changed(playing);
        }
        "seek" => {
            require_source(ui)?;
            if ui.get_mode_index() != 0 {
                return Err("Seeking is only available in Video mode".into());
            }
            let position = args["position"]
                .as_f64()
                .filter(|n| n.is_finite() && (0.0..=1.0).contains(n))
                .ok_or("position must be 0..1")?;
            ui.set_playhead(position as f32);
            ui.invoke_seek(position as f32);
        }
        "add_to_batch" => {
            require_source(ui)?;
            let count = ui.get_tasks().row_count();
            ui.invoke_add_task();
            if ui.get_tasks().row_count() == count {
                return Err(format!(
                    "App did not add a task: {}",
                    ui.get_preview_status()
                ));
            }
        }
        "upscale" | "batch_upscale" => {
            if command == "upscale" {
                ui.invoke_export_current();
            } else {
                ui.invoke_batch_export();
            }
            if !busy(ui) {
                return Err(format!(
                    "App did not schedule an upscale: {}",
                    ui.get_preview_status()
                ));
            }
        }
        "cancel_upscale" => {
            if !busy(ui) {
                return Err("No upscale is running".into());
            }
            ui.invoke_cancel_export();
        }
        "clear_queue" => ui.invoke_clear_queue(),
        "clear_media" => ui.invoke_clear_video(),
        "release_gpu_memory" => ui.invoke_release_gpu_memory(),
        "reveal_output" => {
            let index = args["index"]
                .as_u64()
                .ok_or("index must be a nonnegative task index")? as usize;
            let row = ui
                .get_tasks()
                .row_data(index)
                .ok_or("Task index does not exist")?;
            if !row.done || !std::path::Path::new(row.path.as_str()).is_file() {
                return Err("Output is not complete or no longer exists".into());
            }
            ui.invoke_reveal(row.path);
        }
        "get_logs" => {
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(100);
            if !(1..=500).contains(&limit) {
                return Err("limit must be 1..500".into());
            }
            let model = match args.get("kind").and_then(Value::as_str).unwrap_or("app") {
                "app" => ui.get_log_lines(),
                "mcp" => ui.get_mcp_log_lines(),
                _ => return Err("kind must be app or mcp".into()),
            };
            return Ok(
                json!({"lines":model.iter().take(limit as usize).map(|s|s.to_string()).collect::<Vec<_>>()}),
            );
        }
        _ => return Err(format!("Unknown app command: {command}")),
    }
    Ok(snapshot(ui, source()))
}

fn audit(ui: &MainWindow, text: &str) {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        % 86400;
    let line = format!(
        "{:02}:{:02}:{:02} UTC · {text}",
        stamp / 3600,
        stamp / 60 % 60,
        stamp % 60
    );
    let model = ui.get_mcp_log_lines();
    if let Some(rows) = model.as_any().downcast_ref::<VecModel<SharedString>>() {
        for piece in crate::wrap_log_line(&line, 52).into_iter().rev() {
            rows.insert(0, piece.into());
        }
        while rows.row_count() > 500 {
            rows.remove(rows.row_count() - 1);
        }
    }
    crate::log(&ui.as_weak(), format!("MCP · {text}"));
}

pub fn start(
    ui: &MainWindow,
    import: impl Fn(PathBuf) -> Result<(), String> + 'static,
    source: impl Fn() -> Option<PathBuf> + 'static,
) -> Option<slint::Timer> {
    ui.set_mcp_log_lines(ModelRc::new(VecModel::<SharedString>::default()));
    let server = match mcp_transport::start() {
        Ok(server) => server,
        Err(error) => {
            ui.set_mcp_status(error.clone().into());
            audit(ui, &error);
            return None;
        }
    };
    ui.set_mcp_status(
        format!(
            "Local app control ready at {}. Waiting for an MCP client.",
            mcp_transport::ADDRESS
        )
        .into(),
    );
    audit(ui, "App control ready. No agent connected.");
    let handle = ui.as_weak();
    let sessions = std::cell::RefCell::new(HashMap::<String, (String, Instant)>::new());
    // Cache acknowledgements for this app lifetime so a duplicated request ID
    // never schedules the same action twice. Connection/heartbeat traffic is excluded.
    let replies = std::cell::RefCell::new(HashMap::<(String, String), Value>::new());
    let timer = slint::Timer::default();
    timer.start(slint::TimerMode::Repeated,Duration::from_millis(30),move || {
        let Some(ui)=handle.upgrade() else {return};
        let now=Instant::now();
        let mut sessions=sessions.borrow_mut();
        sessions.retain(|_,(name,last)| {
            if now.duration_since(*last)>LEASE {audit(&ui,&format!("{name} connection expired"));false} else {true}
        });
        for request in server.rx.try_iter().take(8) {
            let mcp_transport::Request{id,session,client,command,args,deadline,reply}=request;
            if Instant::now()>deadline {
                let _=reply.send(json!({"id":id,"ok":false,"error":"Command expired before the app executed it"})); continue;
            }
            let result=match command.as_str() {
                "connect"=>{
                    // The MCP server connects before its host names itself, then reconnects
                    // with the same session once the name is known; audit that rename too.
                    match sessions.get(&session) {
                        None=>audit(&ui,&format!("{client} connected")),
                        Some((name,_)) if *name!=client=>audit(&ui,&format!("{name} identified as {client}")),
                        _=>{}
                    }
                    sessions.insert(session.clone(),(client.clone(),Instant::now()));
                    Ok(json!({"application":"kraken-upscale","pid":std::process::id(),"lease_seconds":12,
                        "project_root":std::env::current_exe().ok().and_then(|p|p.parent().map(|p|p.to_path_buf()))}))
                },
                "heartbeat"=>match sessions.get_mut(&session) {
                    Some((_,last))=>{*last=Instant::now();Ok(json!({"connected":true}))},
                    None=>Err("Session expired; connect again".into()),
                },
                "disconnect"=>{if let Some((name,_))=sessions.remove(&session) {audit(&ui,&format!("{name} disconnected"));} Ok(json!({"connected":false}))},
                _=>{
                    if !sessions.contains_key(&session) {Err("Connect an MCP session first".into())}
                    else {
                        let key=(session.clone(),id.clone());
                        if let Some(previous)=replies.borrow().get(&key) {let _=reply.send(previous.clone());continue;}
                        sessions.get_mut(&session).unwrap().1=Instant::now();
                        // Keep arbitrary argument values out of the audit log.
                        let summary = args.as_object().map(|fields| {
                            fields.keys().take(40).map(|name| {
                                if name == "settings" {
                                    format!("settings: {}", fields[name].as_object()
                                        .map(|settings| settings.keys().take(40).map(|key|key.chars().take(80).collect::<String>()).collect::<Vec<_>>().join(", "))
                                        .unwrap_or_else(|| "invalid".into()))
                                } else { name.chars().take(80).collect() }
                            }).collect::<Vec<_>>().join("; ")
                        }).unwrap_or_else(|| "invalid arguments".into());
                        audit(&ui,&format!("{client} → {command} [{id}] {summary}"));
                        let started=Instant::now();
                        let result=dispatch(&ui,&command,&args,&import,&source);
                        audit(&ui,&format!("{client} ← {command}: {} ({} ms)",
                            result.as_ref().map(|_|"accepted by app".to_owned()).unwrap_or_else(|e|format!("ERROR: {e}")),started.elapsed().as_millis()));
                        result
                    }
                }
            };
            let response=match result {Ok(result)=>json!({"id":id,"ok":true,"result":result}),Err(error)=>json!({"id":id,"ok":false,"error":error})};
            if !matches!(command.as_str(),"connect"|"heartbeat"|"disconnect") {
                let mut cache=replies.borrow_mut();
                // A bounded recent-history cache supplements the bridge's no-retry rule.
                if cache.len()>=256 {cache.clear();}
                cache.insert((session,id),response.clone());
            }
            let _=reply.send(response);
        }
        ui.set_mcp_connected(!sessions.is_empty());
        let mut clients:Vec<_>=sessions.values().map(|(name,_)|name.clone()).collect(); clients.sort();
        ui.set_mcp_clients(if clients.is_empty(){"No agents connected".to_owned()}else{clients.join(" · ")}.into());
        ui.set_mcp_status(format!("Local app control · {} · {} connected agent(s)",mcp_transport::ADDRESS,sessions.len()).into());
    });
    Some(timer)
}

fn settings(ui: &MainWindow) -> Value {
    json!({
        "video_pipeline_index": ui.get_video_pipeline_index(),
        "image_pipeline_index": ui.get_image_pipeline_index(),
        "rtx_vsr_quality_index": ui.get_rtx_vsr_quality_index(),
        "video2x_preset_index": ui.get_video2x_preset_index(),
        "model_index": ui.get_model_index(),
        "generative_model_index": ui.get_generative_model_index(),
        "resolution_index": ui.get_resolution_index(),
        "content_index": ui.get_content_index(),
        "restore_index": ui.get_restore_index(),
        "frame_index": ui.get_frame_index(),
        "crop_position": ui.get_crop_position(),
        "strength": ui.get_strength(),
        "detail_limit": ui.get_detail_limit(),
        "fp16": ui.get_fp16(),
        "auto_tile": ui.get_auto_tile(),
        "detect_fields": ui.get_detect_fields(),
        "match_color": ui.get_match_color(),
        "stability_on": ui.get_stability_on(),
        "lock": ui.get_lock(),
        "cut_detect": ui.get_cut_detect(),
        "interp_on": ui.get_interp_on(),
        "interp_index": ui.get_interp_index(),
        "generative_prompt": ui.get_generative_prompt().to_string(),
        "generative_denoise": ui.get_generative_denoise(),
        "generative_steps_index": ui.get_generative_steps_index(),
        "generative_tile_index": ui.get_generative_tile_index(),
        "generative_quality_finish": ui.get_generative_quality_finish(),
        "video_format_index": ui.get_video_format_index(),
        "video_quality_index": ui.get_video_quality_index(),
        "audio_index": ui.get_audio_index(),
        "image_format_index": ui.get_image_format_index(),
    })
}

fn settings_schema(ui: &MainWindow) -> Value {
    let mut schema = json!({
        "video_pipeline_index": {"type": "integer", "minimum": 0, "maximum": 2, "labels": ["Models", "RTX VSR", "Video2X"]},
        "image_pipeline_index": {"type": "integer", "minimum": 0, "maximum": 2, "labels": ["Lossless", "Generative Refine", "RTX VSR"]},
        "rtx_vsr_quality_index": {"type": "integer", "minimum": 0, "maximum": 3, "labels": ["Low", "Medium", "High", "Ultra"]},
        "video2x_preset_index": {"type": "integer", "minimum": 0},
        "model_index": {"type": "integer", "minimum": 0},
        "generative_model_index": {"type": "integer", "minimum": 0},
        "resolution_index": {"type": "integer", "minimum": 0, "maximum": 6, "labels": ["720p", "1080p", "1440p", "2160p", "2x", "3x", "4x"]},
        "content_index": {"type": "integer", "minimum": 0, "maximum": 7, "labels": ["AI generated video", "Film / movie (24p)", "TV shot on film", "TV shot on video", "Modern digital", "Anime remaster / Blu-ray", "Anime DVD / TV broadcast", "Anime modern digital"]},
        "restore_index": {"type": "integer", "minimum": 0, "maximum": 3, "labels": ["None", "Deblock light", "Deblock DVD/MPEG-2", "Dejpeg"]},
        "frame_index": {"type": "integer", "minimum": 0, "maximum": 1, "labels": ["Original", "16:9 crop"]},
        "crop_position": {"type": "number", "minimum": 0, "maximum": 1},
        "strength": {"type": "number", "minimum": 0, "maximum": 1},
        "detail_limit": {"type": "number", "minimum": 0, "maximum": 1},
        "fp16": {"type": "boolean"},
        "auto_tile": {"type": "boolean"},
        "detect_fields": {"type": "boolean"},
        "match_color": {"type": "boolean"},
        "stability_on": {"type": "boolean"},
        "lock": {"type": "number", "minimum": 0, "maximum": 1},
        "cut_detect": {"type": "number", "minimum": 0, "maximum": 1},
        "interp_on": {"type": "boolean"},
        "interp_index": {"type": "integer", "minimum": 0, "maximum": 2, "labels": [48, 60, 120]},
        "generative_prompt": {"type": "string", "maxLength": 4000},
        "generative_denoise": {"type": "number", "minimum": 0.05, "maximum": 0.4},
        "generative_steps_index": {"type": "integer", "minimum": 0, "maximum": 3, "labels": [8, 15, 20, 28]},
        "generative_tile_index": {"type": "integer", "minimum": 0, "maximum": 2, "labels": [512, 768, 1024]},
        "generative_quality_finish": {"type": "boolean"},
        "video_format_index": {"type": "integer", "minimum": 0, "maximum": 3, "labels": ["MP4 H.264", "MP4 H.265", "MKV H.264", "MKV H.265"]},
        "video_quality_index": {"type": "integer", "minimum": 0, "maximum": 2, "labels": ["Best CQ15", "High CQ19", "Balanced CQ23"]},
        "audio_index": {"type": "integer", "minimum": 0, "maximum": 2, "labels": ["Copy source", "No audio", "AAC 320k (Premiere-friendly)"]},
        "image_format_index": {"type": "integer", "minimum": 0, "maximum": 2, "labels": ["PNG", "JPEG", "WebP"]},
    });
    schema["model_index"]["maximum"] = json!(ui.get_model_names().row_count().saturating_sub(1));
    let presets = strings(ui.get_video2x_preset_names());
    schema["video2x_preset_index"]["maximum"] = json!(presets.len().saturating_sub(1));
    schema["video2x_preset_index"]["labels"] = json!(presets);
    schema["generative_model_index"]["maximum"] = json!(ui
        .get_generative_model_names()
        .row_count()
        .saturating_sub(1));
    schema
}

fn apply_settings(ui: &MainWindow, values: &serde_json::Map<String, Value>) {
    for (key, value) in values {
        match key.as_str() {
            "video_pipeline_index" => ui.set_video_pipeline_index(value.as_i64().unwrap() as i32),
            "image_pipeline_index" => ui.set_image_pipeline_index(value.as_i64().unwrap() as i32),
            "rtx_vsr_quality_index" => ui.set_rtx_vsr_quality_index(value.as_i64().unwrap() as i32),
            "video2x_preset_index" => ui.set_video2x_preset_index(value.as_i64().unwrap() as i32),
            "model_index" => ui.set_model_index(value.as_i64().unwrap() as i32),
            "generative_model_index" => {
                ui.set_generative_model_index(value.as_i64().unwrap() as i32)
            }
            "resolution_index" => ui.set_resolution_index(value.as_i64().unwrap() as i32),
            "content_index" => ui.set_content_index(value.as_i64().unwrap() as i32),
            "restore_index" => ui.set_restore_index(value.as_i64().unwrap() as i32),
            "frame_index" => ui.set_frame_index(value.as_i64().unwrap() as i32),
            "crop_position" => ui.set_crop_position(value.as_f64().unwrap() as f32),
            "strength" => ui.set_strength(value.as_f64().unwrap() as f32),
            "detail_limit" => ui.set_detail_limit(value.as_f64().unwrap() as f32),
            "fp16" => ui.set_fp16(value.as_bool().unwrap()),
            "auto_tile" => ui.set_auto_tile(value.as_bool().unwrap()),
            "detect_fields" => ui.set_detect_fields(value.as_bool().unwrap()),
            "match_color" => ui.set_match_color(value.as_bool().unwrap()),
            "stability_on" => ui.set_stability_on(value.as_bool().unwrap()),
            "lock" => ui.set_lock(value.as_f64().unwrap() as f32),
            "cut_detect" => ui.set_cut_detect(value.as_f64().unwrap() as f32),
            "interp_on" => ui.set_interp_on(value.as_bool().unwrap()),
            "interp_index" => ui.set_interp_index(value.as_i64().unwrap() as i32),
            "generative_prompt" => ui.set_generative_prompt(value.as_str().unwrap().into()),
            "generative_denoise" => ui.set_generative_denoise(value.as_f64().unwrap() as f32),
            "generative_steps_index" => {
                ui.set_generative_steps_index(value.as_i64().unwrap() as i32)
            }
            "generative_tile_index" => ui.set_generative_tile_index(value.as_i64().unwrap() as i32),
            "generative_quality_finish" => {
                ui.set_generative_quality_finish(value.as_bool().unwrap())
            }
            "video_format_index" => ui.set_video_format_index(value.as_i64().unwrap() as i32),
            "video_quality_index" => ui.set_video_quality_index(value.as_i64().unwrap() as i32),
            "audio_index" => ui.set_audio_index(value.as_i64().unwrap() as i32),
            "image_format_index" => ui.set_image_format_index(value.as_i64().unwrap() as i32),
            _ => unreachable!("validated field"),
        }
    }
}

fn view(ui: &MainWindow) -> Value {
    json!({
        "view_mode": ui.get_view_mode(),
        "split": ui.get_split(),
        "zoom": ui.get_zoom(),
        "pan_x": ui.get_pan_x(),
        "pan_y": ui.get_pan_y(),
        "src_open": ui.get_src_open(),
        "det_open": ui.get_det_open(),
        "stb_open": ui.get_stb_open(),
        "fi_open": ui.get_fi_open(),
        "col_open": ui.get_col_open(),
    })
}

fn view_schema(ui: &MainWindow) -> Value {
    let schema = json!({
        "view_mode": {"type": "integer", "minimum": 0, "maximum": 1, "labels": ["Split", "Side by side"]},
        "split": {"type": "number", "minimum": 0, "maximum": 1},
        "zoom": {"type": "number", "minimum": 1, "maximum": 32},
        "pan_x": {"type": "number", "minimum": -100000, "maximum": 100000},
        "pan_y": {"type": "number", "minimum": -100000, "maximum": 100000},
        "src_open": {"type": "boolean"},
        "det_open": {"type": "boolean"},
        "stb_open": {"type": "boolean"},
        "fi_open": {"type": "boolean"},
        "col_open": {"type": "boolean"},
    });
    let _ = ui;
    schema
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_controls_before_applying_any_setting() {
        let schema=json!({"quality":{"type":"integer","minimum":0,"maximum":3},"enabled":{"type":"boolean"}});
        for input in [json!({"quality":4}), json!({"quality":1.5}), json!({"enabled":1}),
            json!({"quality":2,"typo":true}), json!({}), json!([])] {
            assert!(validate(&input,&schema).is_err(),"{input}");
        }
        assert!(validate(&json!({"quality":3,"enabled":true}),&schema).is_ok());
    }

    #[test]
    fn bounds_text_and_floating_point_settings() {
        let schema=json!({"prompt":{"type":"string"},"strength":{"type":"number","minimum":0.0,"maximum":1.0}});
        assert!(validate(&json!({"prompt":"x".repeat(4001)}),&schema).is_err());
        assert!(validate(&json!({"prompt":"bad\0text"}),&schema).is_err());
        assert!(validate(&json!({"strength":-0.01}),&schema).is_err());
        assert!(validate(&json!({"strength":0.5,"prompt":"Keep the source detail"}),&schema).is_ok());
    }
}

fn apply_view(ui: &MainWindow, values: &serde_json::Map<String, Value>) {
    for (key, value) in values {
        match key.as_str() {
            "view_mode" => ui.set_view_mode(value.as_i64().unwrap() as i32),
            "split" => ui.set_split(value.as_f64().unwrap() as f32),
            "zoom" => ui.set_zoom(value.as_f64().unwrap() as f32),
            "pan_x" => ui.set_pan_x(value.as_f64().unwrap() as f32),
            "pan_y" => ui.set_pan_y(value.as_f64().unwrap() as f32),
            "src_open" => ui.set_src_open(value.as_bool().unwrap()),
            "det_open" => ui.set_det_open(value.as_bool().unwrap()),
            "stb_open" => ui.set_stb_open(value.as_bool().unwrap()),
            "fi_open" => ui.set_fi_open(value.as_bool().unwrap()),
            "col_open" => ui.set_col_open(value.as_bool().unwrap()),
            _ => unreachable!("validated field"),
        }
    }
}
