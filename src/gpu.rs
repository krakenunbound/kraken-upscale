//! GPU identification and live telemetry.
//!
//! Shells out to nvidia-smi rather than linking NVML: this is a status
//! readout, and a missing driver should degrade to a label rather than a link
//! error at startup.

use std::process::Command;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub struct Probe {
    pub name: String,
    pub runtime: String,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub device: String,
    pub usage_pct: u32,
    pub temp_c: u32,
    pub clock_mhz: u32,
    pub clock_max_mhz: u32,
    pub power_w: f32,
    pub power_max_w: f32,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub driver: String,
}

impl Stats {
    pub fn vram_used_gb(&self) -> f32 {
        (self.vram_total_mb.saturating_sub(self.vram_free_mb)) as f32 / 1024.0
    }
    pub fn vram_total_gb(&self) -> f32 {
        self.vram_total_mb as f32 / 1024.0
    }
    pub fn vram_free_gb(&self) -> f32 {
        self.vram_free_mb as f32 / 1024.0
    }
}

fn smi(args: &[&str]) -> Option<String> {
    let mut cmd = Command::new("nvidia-smi");
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let out = cmd.args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// One-shot read for the title-bar pills.
pub fn probe() -> Probe {
    match stats() {
        Some(s) => Probe {
            name: format!("{}  ·  {:.0} GB", short_name(&s.device), s.vram_total_gb()),
            runtime: "RUNTIME OK".into(),
        },
        None => Probe {
            name: "NO CUDA DEVICE".into(),
            runtime: "CPU FALLBACK".into(),
        },
    }
}

/// Full telemetry snapshot.
pub fn stats() -> Option<Stats> {
    let text = smi(&[
        "--query-gpu=name,utilization.gpu,temperature.gpu,clocks.current.graphics,\
         clocks.max.graphics,power.draw,power.limit,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])?;

    let line = text.lines().next()?;
    let f: Vec<&str> = line.split(',').map(str::trim).collect();
    if f.len() < 10 {
        return None;
    }

    Some(Stats {
        device: f[0].to_string(),
        usage_pct: f[1].parse().unwrap_or(0),
        temp_c: f[2].parse().unwrap_or(0),
        clock_mhz: f[3].parse().unwrap_or(0),
        clock_max_mhz: f[4].parse().unwrap_or(0),
        power_w: f[5].parse().unwrap_or(0.0),
        power_max_w: f[6].parse().unwrap_or(0.0),
        vram_total_mb: f[7].parse().unwrap_or(0),
        vram_free_mb: f[8].parse().unwrap_or(0),
        driver: f[9].to_string(),
    })
}

/// "NVIDIA GeForce RTX 3090" -> "RTX 3090"
pub fn short_name(full: &str) -> String {
    full.split_whitespace()
        .skip_while(|w| matches!(*w, "NVIDIA" | "GeForce"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Versions of everything the pipeline depends on.
#[derive(Debug, Clone, Default)]
pub struct Runtime {
    pub app: String,
    pub ort: String,
    pub onnxruntime: String,
    pub ffmpeg: String,
    pub cuda: String,
    pub cudnn: String,
}

pub fn runtime(cuda_dir: Option<&std::path::Path>) -> Runtime {
    Runtime {
        app: env!("CARGO_PKG_VERSION").to_string(),
        // ONNX Runtime's own version string is only reachable inside `ort`'s
        // loader, so these are the compile-time known values.
        ort: "2.0.0-rc.13".into(),
        onnxruntime: "1.28".into(),
        ffmpeg: ffmpeg_version().unwrap_or_else(|| "not found".into()),
        // Derived from the DLL sonames present rather than guessed: CUDA and
        // cuDNN carry their major version in the filename.
        cuda: cuda_dir
            .map(|d| {
                if d.join("cublasLt64_13.dll").is_file() {
                    "13.x".to_string()
                } else if d.join("cublasLt64_12.dll").is_file() {
                    "12.x".to_string()
                } else {
                    "unknown".to_string()
                }
            })
            .unwrap_or_else(|| "not found".into()),
        cudnn: cuda_dir
            .map(|d| {
                if d.join("cudnn64_9.dll").is_file() {
                    "9.x".to_string()
                } else if d.join("cudnn64_8.dll").is_file() {
                    "8.x".to_string()
                } else {
                    "unknown".to_string()
                }
            })
            .unwrap_or_else(|| "not found".into()),
    }
}

fn ffmpeg_version() -> Option<String> {
    let mut cmd = Command::new(crate::media::ffmpeg());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let out = cmd.args(["-hide_banner", "-version"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let first = text.lines().next()?;
    // "ffmpeg version n8.1.2-34-g9b6c8969e0-20260806 Copyright ..."
    first
        .split_whitespace()
        .nth(2)
        .map(|v| v.split('-').next().unwrap_or(v).to_string())
}
