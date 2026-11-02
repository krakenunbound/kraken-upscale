//! Model and runtime catalog behind the Models tab.
//!
//! Each entry knows where it comes from, which files prove it is installed,
//! and where those files belong, so a user never has to work out folders by
//! hand. Downloads use the `curl.exe` and `tar.exe` that ship with Windows 10
//! and 11 (`tar` is bsdtar, which also reads 7z), so Kraken carries no HTTP or
//! archive code of its own.
//!
//! Some items cannot be fetched automatically: NVIDIA's RTX VSR runtime is
//! distributed by NVIDIA under its own terms, NanoVSR only publishes PyTorch
//! weights, and FLUX needs a licence acceptance. Those carry instructions.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub enum Source {
    /// A single file saved as `installs[0]`.
    File { url: &'static str },
    /// Archive members, each saved under a new name in the entry's folder.
    Archive {
        url: &'static str,
        members: &'static [(&'static str, &'static str)],
    },
    /// A whole archive unpacked into the folder, minus build-only paths.
    Bundle {
        url: &'static str,
        skip: &'static [&'static str],
    },
    /// Not downloadable here; `steps` explains what to do.
    Manual { steps: &'static str },
}

/// Metadata appended to the folder's `models.json`, so the picker shows a
/// proper name and scale instead of guessing from the file name.
pub struct Manifest {
    pub name: &'static str,
    pub scale: u32,
    pub architecture: &'static str,
}

pub struct Entry {
    pub name: &'static str,
    pub group: &'static str,
    pub summary: &'static str,
    /// Relative to the folder that holds the executable.
    pub folder: &'static str,
    /// Files (relative to `folder`) whose presence means "installed".
    pub installs: &'static [&'static str],
    /// Download size in bytes, for progress and the size label.
    pub bytes: u64,
    pub license: &'static str,
    pub page: &'static str,
    pub source: Source,
    pub manifest: Option<Manifest>,
}

const CUGAN_PRO: &str =
    "https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/cugan-pro_v3.7z";
const ANISD: &str = "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD/";

pub static CATALOG: &[Entry] = &[
    Entry {
        name: "Real-CUGAN Pro Conservative 3x",
        group: "VIDEO · ANIMATION",
        summary: "Faithful cel-animation upscale with light cleanup, native 3x. Best fit for 1080p → 4K-class anime; no final resize.",
        folder: "models/video-frame",
        installs: &["real-cugan-pro-conservative-3x.onnx"],
        bytes: 28_064_239,
        license: "MIT",
        page: "https://github.com/bilibili/ailab/tree/main/Real-CUGAN",
        source: Source::Archive {
            url: CUGAN_PRO,
            members: &[("cugan/pro-conservative-up3x.onnx", "real-cugan-pro-conservative-3x.onnx")],
        },
        manifest: Some(Manifest { name: "Real-CUGAN Pro Conservative 3x", scale: 3, architecture: "Real-CUGAN Pro" }),
    },
    Entry {
        name: "Real-CUGAN Pro Conservative 2x",
        group: "VIDEO · ANIMATION",
        summary: "Same faithful model at 2x. For 720p → 1440p, or SD sources.",
        folder: "models/video-frame",
        installs: &["real-cugan-pro-conservative-2x.onnx"],
        bytes: 28_064_239,
        license: "MIT",
        page: "https://github.com/bilibili/ailab/tree/main/Real-CUGAN",
        source: Source::Archive {
            url: CUGAN_PRO,
            members: &[("cugan/pro-conservative-up2x.onnx", "real-cugan-pro-conservative-2x.onnx")],
        },
        manifest: Some(Manifest { name: "Real-CUGAN Pro Conservative", scale: 2, architecture: "Real-CUGAN Pro" }),
    },
    Entry {
        name: "Real-CUGAN Pro Denoise 3",
        group: "VIDEO · ANIMATION",
        summary: "Strong denoise for noisy DVD, VHS or heavily compressed animation. Can thicken lines.",
        folder: "models/video-frame",
        installs: &["real-cugan-pro-denoise3x-2x.onnx"],
        bytes: 28_064_239,
        license: "MIT",
        page: "https://github.com/bilibili/ailab/tree/main/Real-CUGAN",
        source: Source::Archive {
            url: CUGAN_PRO,
            members: &[("cugan/pro-denoise3x-up2x.onnx", "real-cugan-pro-denoise3x-2x.onnx")],
        },
        manifest: Some(Manifest { name: "Real-CUGAN Pro Denoise 3", scale: 2, architecture: "Real-CUGAN Pro" }),
    },
    Entry {
        name: "Real-ESRGAN AnimeVideo v3",
        group: "VIDEO · ANIMATION",
        summary: "Compact video-tuned anime model, 4x. Faster than Real-CUGAN, slightly softer.",
        folder: "models/video-frame",
        installs: &["realesr-animevideov3.onnx"],
        bytes: 2_306_369,
        license: "BSD-3-Clause",
        page: "https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md",
        source: Source::Archive {
            url: "https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/RealESRGANv3_v1.7z",
            members: &[("RealESRGANv2/realesr-animevideov3.onnx", "realesr-animevideov3.onnx")],
        },
        manifest: Some(Manifest { name: "Real-ESRGAN AnimeVideo v3", scale: 4, architecture: "SRVGGNetCompact" }),
    },
    Entry {
        name: "AniSD Cel SPAN",
        group: "VIDEO · ANIMATION",
        summary: "2x for standard-definition cel animation with flat colour.",
        folder: "models/video-frame",
        installs: &["2x-AniSD-Cel-SPAN.onnx"],
        bytes: 1_656_315,
        license: "See page",
        page: "https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD",
        source: Source::File {
            url: "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD/2x_AniSD_AC_G6i2b_SPAN_190K_fp32.onnx",
        },
        manifest: Some(Manifest { name: "AniSD Cel SPAN", scale: 2, architecture: "SPAN" }),
    },
    Entry {
        name: "AniSD General SPAN",
        group: "VIDEO · ANIMATION",
        summary: "2x for general standard-definition animation.",
        folder: "models/video-frame",
        installs: &["2x-AniSD-General-SPAN.onnx"],
        bytes: 1_656_315,
        license: "See page",
        page: "https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD",
        source: Source::File {
            url: "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD/2x_AniSD_G6i1_SPAN_215K_fp32.onnx",
        },
        manifest: Some(Manifest { name: "AniSD General SPAN", scale: 2, architecture: "SPAN" }),
    },
    Entry {
        name: "ModernSpanimation V1",
        group: "VIDEO · ANIMATION",
        summary: "2x for modern digital and AI-generated 2D animation.",
        folder: "models/video-frame",
        installs: &["2x-ModernSpanimationV1.onnx"],
        bytes: 4_039_210,
        license: "See page",
        page: "https://github.com/TNTwise/Models/releases/tag/2x_ModernSpanimationV1",
        source: Source::Archive {
            url: "https://github.com/TNTwise/Models/releases/download/2x_ModernSpanimationV1/ONNX.zip",
            members: &[("ONNX/2x_ModernSpanimationV1_fp32_op17.onnx", "2x-ModernSpanimationV1.onnx")],
        },
        manifest: Some(Manifest { name: "ModernSpanimation V1", scale: 2, architecture: "SPAN" }),
    },
    Entry {
        name: "Real-ESRGAN General x4v3",
        group: "VIDEO · LIVE ACTION",
        summary: "Compact general model for live-action and AI video, 4x.",
        folder: "models/video-frame",
        installs: &["realesr-general-x4v3.onnx"],
        bytes: 4_866_417,
        license: "BSD-3-Clause",
        page: "https://huggingface.co/CoderViking/realesr-general-x4v3-onnx",
        source: Source::File {
            url: "https://huggingface.co/CoderViking/realesr-general-x4v3-onnx/resolve/main/realesr-general-x4v3.onnx",
        },
        manifest: Some(Manifest { name: "Real-ESRGAN General x4v3", scale: 4, architecture: "SRVGGNetCompact" }),
    },
    Entry {
        name: "OpenProteus Compact",
        group: "VIDEO · LIVE ACTION",
        summary: "Faithful 2x for live-action HD and Full HD.",
        folder: "models/video-frame",
        installs: &["2x-OpenProteus.onnx"],
        bytes: 2_412_381,
        license: "See page",
        page: "https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus",
        source: Source::File {
            url: "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K_fp32.onnx",
        },
        manifest: Some(Manifest { name: "OpenProteus Compact", scale: 2, architecture: "Compact" }),
    },
    Entry {
        name: "NanoVSR 644k (multi-frame)",
        group: "VIDEO · MULTI-FRAME",
        summary: "Genuine 5-frame video model, 4x. Published only as PyTorch weights.",
        folder: "models/video",
        installs: &["NanoVSR-644k-T5.onnx", "NanoVSR-644k-T5.onnx.data"],
        bytes: 3_001_663,
        license: "MIT",
        page: "https://github.com/filippawlicki/nanovsr/releases/tag/v1.0",
        source: Source::Manual {
            steps: "The project ships nanovsr_644k.pth only. Export it to ONNX with the project's code (5-frame window, opset 18) and save NanoVSR-644k-T5.onnx plus its .onnx.data into the folder below. Details: models/video/README.md.",
        },
        manifest: None,
    },
    Entry {
        name: "RIFE frame interpolation",
        group: "MOTION",
        summary: "Required for 48 / 60 / 120 fps interpolation in the Motion tab.",
        folder: "models/interpolation",
        installs: &["RIFE_fp32_timestep.onnx"],
        bytes: 21_604_567,
        license: "MIT",
        page: "https://huggingface.co/walterlow/RIFE_fp32_timestep",
        source: Source::File {
            url: "https://huggingface.co/walterlow/RIFE_fp32_timestep/resolve/main/RIFE_fp32_timestep.onnx",
        },
        manifest: None,
    },
    Entry {
        name: "4x-UltraSharp",
        group: "IMAGES",
        summary: "Popular sharp general-purpose still-image model.",
        folder: "models",
        installs: &["4x-UltraSharp.pth"],
        bytes: 66_961_958,
        license: "CC BY-NC-SA 4.0",
        page: "https://huggingface.co/Kim2091/UltraSharp",
        source: Source::File {
            url: "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth",
        },
        manifest: None,
    },
    Entry {
        name: "4x-AnimeSharp",
        group: "IMAGES",
        summary: "Still-image model for anime art and screenshots.",
        folder: "models",
        installs: &["4x-AnimeSharp.pth"],
        bytes: 67_010_245,
        license: "CC BY-NC-SA 4.0",
        page: "https://huggingface.co/Kim2091/AnimeSharp",
        source: Source::File {
            url: "https://huggingface.co/Kim2091/AnimeSharp/resolve/main/4x-AnimeSharp.pth",
        },
        manifest: None,
    },
    Entry {
        name: "RealESRGAN x4plus",
        group: "IMAGES",
        summary: "The original general Real-ESRGAN photo model, 4x.",
        folder: "models",
        installs: &["RealESRGAN_x4plus.pth"],
        bytes: 67_040_989,
        license: "BSD-3-Clause",
        page: "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0",
        source: Source::File {
            url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        },
        manifest: None,
    },
    Entry {
        name: "RealESRGAN x2plus",
        group: "IMAGES",
        summary: "General Real-ESRGAN photo model, 2x.",
        folder: "models",
        installs: &["RealESRGAN_x2plus.pth"],
        bytes: 67_061_725,
        license: "BSD-3-Clause",
        page: "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1",
        source: Source::File {
            url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        },
        manifest: None,
    },
    Entry {
        name: "Video2X 6.4.0 runtime",
        group: "RUNTIMES",
        summary: "Enables the Video2X pipeline: Real-ESRGAN, Real-CUGAN and Anime4K on Vulkan.",
        folder: "video2x",
        installs: &["video2x.exe", "libvideo2x.dll"],
        bytes: 196_989_277,
        license: "AGPL-3.0",
        page: "https://github.com/k4yt3x/video2x/releases/tag/6.4.0",
        source: Source::Bundle {
            url: "https://github.com/k4yt3x/video2x/releases/download/6.4.0/video2x-windows-amd64.zip",
            skip: &["include", "cmake", "pkgconfig", ".lib"],
        },
        manifest: None,
    },
    Entry {
        name: "NVIDIA RTX VSR runtime",
        group: "RUNTIMES",
        summary: "Enables the RTX VSR pipeline. Needs an RTX GPU and driver 570.65 or newer.",
        folder: "rtx-vsr",
        installs: &["NVVideoEffects.dll", "nvVFXVideoSuperRes.dll", "nvngx_vsr.dll"],
        bytes: 0,
        license: "NVIDIA proprietary",
        page: "https://developer.nvidia.com/maxine",
        source: Source::Manual {
            steps: "NVIDIA distributes this runtime itself. Get the Video Effects SDK (Maxine) from NVIDIA, accept its licence, then copy NVCVImage.dll, NVVideoEffects.dll, nvVFXVideoSuperRes.dll, nvngx_vsr.dll, nvngxruntime.dll, the npp*64_12.dll files and nvinfer_10 / nvinfer_plugin_10 / nvonnxparser_10.dll into the folder below. See RTX-VSR.md.",
        },
        manifest: None,
    },
    Entry {
        name: "FLUX.1 Dev Q6 (Generative Refine)",
        group: "RUNTIMES",
        summary: "Images · Generative refine. About 15 GB with its encoders. Non-commercial licence.",
        folder: "models/generative/FLUX1-Dev",
        installs: &["flux1-dev-Q6_K.gguf"],
        bytes: 0,
        license: "FLUX.1 Dev non-commercial",
        page: "https://huggingface.co/city96/FLUX.1-dev-gguf",
        source: Source::Manual {
            steps: "Download flux1-dev-Q6_K.gguf from the page, plus the FLUX VAE (ae.safetensors), clip_l.safetensors and a T5-XXL encoder, into the folder below, matching models/generative/flux1-dev.json. The stable-diffusion.cpp CUDA runtime goes in native-diffusion/. Details: models/generative/README.md.",
        },
        manifest: None,
    },
];

/// The folder that holds `models/`, `video2x/` and `rtx-vsr/`.
pub fn app_root() -> PathBuf {
    crate::upscale::models_dir()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

impl Entry {
    pub fn folder_path(&self) -> PathBuf {
        app_root().join(self.folder)
    }

    pub fn installed(&self) -> bool {
        let folder = self.folder_path();
        self.installs.iter().all(|file| folder.join(file).is_file())
    }

    pub fn downloadable(&self) -> bool {
        !matches!(self.source, Source::Manual { .. })
    }

    pub fn steps(&self) -> &'static str {
        match self.source {
            Source::Manual { steps } => steps,
            _ => "",
        }
    }

    pub fn size_label(&self) -> String {
        match self.bytes {
            0 => String::new(),
            b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
            b if b >= 1 << 20 => format!("{:.0} MB", b as f64 / (1u64 << 20) as f64),
            b => format!("{} KB", (b / 1024).max(1)),
        }
    }
}

fn system_tool(name: &str) -> PathBuf {
    // Prefer the Windows copies by absolute path: a Git or MSYS `tar` earlier
    // on PATH cannot read 7z.
    let system = std::env::var_os("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
        .join("System32")
        .join(name);
    if system.is_file() {
        system
    } else {
        PathBuf::from(name)
    }
}

fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped())
}

fn run(cmd: &mut Command, what: &str) -> Result<(), String> {
    let out = quiet(cmd)
        .output()
        .map_err(|error| format!("could not start {what}: {error}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{what} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

/// Downloads `url` to `dest`, reporting bytes as the file grows.
fn download(
    url: &str,
    dest: &Path,
    total: u64,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<(), String> {
    let mut child = quiet(Command::new(system_tool("curl.exe")).args([
        "-L", "--fail", "--retry", "3", "--silent", "--show-error", "-o",
    ]))
    .arg(dest)
    .arg(url)
    .spawn()
    .map_err(|error| format!("could not start curl: {error}"))?;
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("cancelled".into());
        }
        let done = std::fs::metadata(dest).map(|meta| meta.len()).unwrap_or(0);
        on_progress(done, total);
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(_)) => {
                let mut text = String::new();
                if let Some(mut err) = child.stderr.take() {
                    let _ = std::io::Read::read_to_string(&mut err, &mut text);
                }
                return Err(format!("download failed: {}", text.trim()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(250)),
            Err(error) => return Err(format!("download failed: {error}")),
        }
    }
}

fn move_file(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    if std::fs::rename(from, to).is_err() {
        std::fs::copy(from, to).map_err(|error| format!("could not place {}: {error}", to.display()))?;
        let _ = std::fs::remove_file(from);
    }
    Ok(())
}

fn copy_tree(from: &Path, to: &Path, skip: &[&str]) -> Result<(), String> {
    for item in std::fs::read_dir(from).map_err(|error| error.to_string())? {
        let item = item.map_err(|error| error.to_string())?;
        let name = item.file_name().to_string_lossy().into_owned();
        if skip.iter().any(|s| name == *s || (s.starts_with('.') && name.ends_with(s))) {
            continue;
        }
        let target = to.join(&name);
        if item.path().is_dir() {
            std::fs::create_dir_all(&target).map_err(|error| error.to_string())?;
            copy_tree(&item.path(), &target, skip)?;
        } else {
            move_file(&item.path(), &target)?;
        }
    }
    Ok(())
}

fn add_manifest(folder: &Path, file: &str, manifest: &Manifest) -> Result<(), String> {
    let path = folder.join("models.json");
    let mut entries: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    if entries.iter().any(|entry| entry["file"] == file) {
        return Ok(());
    }
    entries.push(serde_json::json!({
        "name": manifest.name,
        "file": file,
        "scale": manifest.scale,
        "architecture": manifest.architecture,
        "temporal_frames": 1,
        "video_tuned": true,
    }));
    let text = serde_json::to_string_pretty(&entries).map_err(|error| error.to_string())?;
    std::fs::write(&path, text).map_err(|error| format!("could not update models.json: {error}"))
}

/// Downloads and installs one catalog entry into its folder.
pub fn install(
    entry: &Entry,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<(), String> {
    let folder = entry.folder_path();
    std::fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let work = app_root()
        .join(".downloads")
        .join(entry.installs[0].replace(['/', '\\'], "_"));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).map_err(|error| error.to_string())?;

    let result = (|| -> Result<(), String> {
        match &entry.source {
            Source::Manual { .. } => Err("this item is installed by hand; see its steps".into()),
            Source::File { url } => {
                let part = work.join("download.part");
                download(url, &part, entry.bytes, cancel, on_progress)?;
                move_file(&part, &folder.join(entry.installs[0]))
            }
            Source::Archive { url, members } => {
                let archive = work.join(
                    url.rsplit('/').next().filter(|name| !name.is_empty()).unwrap_or("archive"),
                );
                download(url, &archive, entry.bytes, cancel, on_progress)?;
                let unpacked = work.join("unpacked");
                std::fs::create_dir_all(&unpacked).map_err(|error| error.to_string())?;
                let mut tar = Command::new(system_tool("tar.exe"));
                tar.arg("-xf").arg(&archive).arg("-C").arg(&unpacked);
                for (member, _) in members.iter() {
                    tar.arg(member);
                }
                run(&mut tar, "unpacking")?;
                for (member, dest) in members.iter() {
                    move_file(&unpacked.join(member), &folder.join(dest))?;
                }
                Ok(())
            }
            Source::Bundle { url, skip } => {
                let archive = work.join("bundle.zip");
                download(url, &archive, entry.bytes, cancel, on_progress)?;
                let unpacked = work.join("unpacked");
                std::fs::create_dir_all(&unpacked).map_err(|error| error.to_string())?;
                run(
                    Command::new(system_tool("tar.exe"))
                        .arg("-xf")
                        .arg(&archive)
                        .arg("-C")
                        .arg(&unpacked),
                    "unpacking",
                )?;
                copy_tree(&unpacked, &folder, skip)
            }
        }?;
        if let Some(manifest) = &entry.manifest {
            add_manifest(&folder, entry.installs[0], manifest)?;
        }
        if entry.installed() {
            Ok(())
        } else {
            Err("download finished but the expected files are missing".into())
        }
    })();
    let _ = std::fs::remove_dir_all(&work);
    result
}

/// Opens a web page in the default browser.
pub fn open_page(url: &str) {
    #[cfg(windows)]
    let _ = Command::new("rundll32")
        .args(["url.dll,FileProtocolHandler", url])
        .creation_flags(CREATE_NO_WINDOW)
        .spawn();
}

/// Opens an entry's folder in Explorer, creating it first so it always works.
pub fn open_folder(entry: &Entry) {
    let folder = entry.folder_path();
    let _ = std::fs::create_dir_all(&folder);
    let _ = Command::new("explorer").arg(&folder).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "downloads ~28 MB from GitHub into models/video-frame"]
    fn downloads_and_installs_cugan_3x() {
        let entry = CATALOG
            .iter()
            .find(|e| e.name == "Real-CUGAN Pro Conservative 3x")
            .unwrap();
        let mut last = 0;
        install(entry, &AtomicBool::new(false), &mut |done, _| last = done).unwrap();
        assert!(entry.installed());
        assert_eq!(last, entry.bytes);
    }

    #[test]
    fn every_entry_is_complete() {
        for entry in CATALOG {
            assert!(!entry.installs.is_empty(), "{}", entry.name);
            assert!(entry.page.starts_with("https://"), "{}", entry.name);
            match &entry.source {
                Source::Manual { steps } => assert!(!steps.is_empty()),
                _ => assert!(entry.bytes > 0, "{} needs a size", entry.name),
            }
        }
    }
}
