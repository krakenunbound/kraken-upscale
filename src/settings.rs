//! Persisted UI settings.
//!
//! Stored under `%APPDATA%\KrakenUpscale\settings.json`, not next to the
//! executable — a Program Files install is not writable by the user.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Stored **by name**, not index: the model list changes as files are
    /// added or removed, and an index would silently select a different model.
    pub model: String,
    /// 0 = Video, 1 = Image. Kept as an integer to match the UI segment.
    pub mode_index: i32,
    pub video_model: String,
    pub image_model: String,
    /// 0 = faithful model-only upscale, 1 = faithful + generative tile refine.
    pub image_pipeline_index: i32,
    pub generative_model: String,
    pub generative_prompt: String,
    pub generative_denoise: f32,
    pub generative_steps: i32,
    pub generative_tile_index: i32,
    #[serde(default = "default_true")]
    pub generative_quality_finish: bool,
    pub scale_index: i32,
    pub strength: f32,
    pub detail_limit: f32,
    pub fp16: bool,
    #[serde(default = "default_true")]
    pub auto_tile: bool,

    pub content_index: i32,
    pub restore_index: i32,
    pub detect_fields: bool,
    pub preserve_grain: bool,
    pub match_color: bool,

    pub stability: bool,
    pub lock: f32,
    pub cut_detect: f32,

    pub interp_enabled: bool,
    pub interp_index: i32,

    pub dock_panel: i32,
    pub view_mode: i32,

    pub codec: String,
    pub cq: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: String::new(),
            mode_index: 0,
            video_model: String::new(),
            image_model: String::new(),
            image_pipeline_index: 0,
            generative_model: String::new(),
            generative_prompt: "high detail, faithful textures, preserve composition".into(),
            generative_denoise: 0.10,
            generative_steps: 15,
            generative_tile_index: 1,
            generative_quality_finish: true,
            scale_index: 0,
            strength: 1.0,
            detail_limit: 1.0,
            fp16: true,
            auto_tile: true,

            content_index: 0,
            restore_index: 0,
            detect_fields: false,
            preserve_grain: false,
            match_color: true,

            stability: true,
            lock: 0.75,
            cut_detect: 0.4,

            interp_enabled: false,
            interp_index: 1,

            dock_panel: 0,
            view_mode: 0,

            codec: "h264_nvenc".into(),
            cq: 19,
        }
    }
}

fn default_true() -> bool {
    true
}

pub fn path() -> PathBuf {
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("KrakenUpscale").join("settings.json")
}

pub fn load() -> Settings {
    match std::fs::read_to_string(path()) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            eprintln!("settings unreadable, using defaults: {e}");
            Settings::default()
        }),
        Err(_) => Settings::default(),
    }
}

pub fn save(s: &Settings) {
    let p = path();
    if let Some(dir) = p.parent() {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("could not create settings directory: {e}");
            return;
        }
    }
    match serde_json::to_string_pretty(s) {
        Ok(text) => {
            if let Err(e) = std::fs::write(&p, text) {
                eprintln!("could not write settings: {e}");
            }
        }
        Err(e) => eprintln!("could not serialise settings: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_settings_default_to_video_mode() {
        let settings: Settings =
            serde_json::from_str(r#"{"model":"4x-UltraSharp"}"#).expect("read legacy settings");
        assert_eq!(settings.mode_index, 0);
        assert!(settings.video_model.is_empty());
        assert!(settings.image_model.is_empty());
        assert_eq!(settings.image_pipeline_index, 0);
    }

    #[test]
    fn image_mode_and_per_mode_models_round_trip() {
        let settings = Settings {
            mode_index: 1,
            video_model: "NanoVSR-644k (video)".into(),
            image_model: "4x_foolhardy_Remacri".into(),
            image_pipeline_index: 1,
            generative_model: "FLUX.1 Dev".into(),
            generative_denoise: 0.18,
            ..Settings::default()
        };
        let text = serde_json::to_string(&settings).expect("serialize settings");
        let restored: Settings = serde_json::from_str(&text).expect("restore settings");
        assert_eq!(restored.mode_index, 1);
        assert_eq!(restored.video_model, settings.video_model);
        assert_eq!(restored.image_model, settings.image_model);
        assert_eq!(restored.image_pipeline_index, 1);
        assert_eq!(restored.generative_model, "FLUX.1 Dev");
        assert_eq!(restored.generative_denoise, 0.18);
    }
}
