//! Still-image input and lossless export.
//!
//! Image mode intentionally reuses the same native ONNX detail engine as
//! video. The difference is the source/output contract: one RGB frame enters,
//! temporal stages are invalid, and one lossless PNG leaves.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use crate::media::Frame;
use crate::upscale::{self, Engine, ModelEntry};

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub codec: String,
    pub pix_fmt: String,
    pub has_alpha: bool,
}

impl ImageInfo {
    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub fn specs(&self) -> String {
        format!("{} × {}  ·  {}", self.width, self.height, self.codec)
    }
}

#[derive(Clone)]
pub struct ExportSpec {
    pub image: ImageInfo,
    pub model: ModelEntry,
    pub target_scale: u32,
    pub strength: f32,
    pub detail_limit: f32,
    pub match_color: bool,
    pub fp16: bool,
    pub tile: u32,
    pub out_path: PathBuf,
}

impl ExportSpec {
    pub fn output_size(&self) -> (u32, u32) {
        (
            self.image.width.saturating_mul(self.target_scale),
            self.image.height.saturating_mul(self.target_scale),
        )
    }
}

pub fn is_supported(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff")
    )
}

fn pixel_format_has_alpha(format: &str) -> bool {
    let format = format.to_ascii_lowercase();
    format.starts_with("rgba")
        || format.starts_with("bgra")
        || format.starts_with("argb")
        || format.starts_with("abgr")
        || format.starts_with("yuva")
        || format.starts_with("gbrap")
        || format.starts_with("ya")
}

fn quiet(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    command.creation_flags(CREATE_NO_WINDOW);
    command
}

pub fn probe(path: &Path) -> Result<ImageInfo, String> {
    if !is_supported(path) {
        return Err("unsupported image type; use PNG, JPEG, WebP, BMP, or TIFF".into());
    }

    let output = quiet(&mut Command::new(crate::media::ffprobe()))
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .map_err(|error| format!("could not inspect image: {error}"))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let field = |key: &str| -> Option<String> {
        text.lines()
            .find_map(|line| line.split_once('=').filter(|(name, _)| *name == key))
            .map(|(_, value)| value.trim().to_string())
    };
    let width = field("width")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let height = field("height")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if width == 0 || height == 0 {
        return Err("image has no readable dimensions".into());
    }

    let pix_fmt = field("pix_fmt").unwrap_or_else(|| "unknown".into());
    Ok(ImageInfo {
        path: path.to_path_buf(),
        width,
        height,
        codec: field("codec_name").unwrap_or_else(|| "image".into()),
        has_alpha: pixel_format_has_alpha(&pix_fmt),
        pix_fmt,
    })
}

pub fn decode(info: &ImageInfo) -> Result<Frame, String> {
    let output = quiet(&mut Command::new(crate::media::ffmpeg()))
        .args(["-v", "error", "-i"])
        .arg(&info.path)
        .args([
            "-frames:v",
            "1",
            "-vf",
            "setsar=1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .map_err(|error| format!("could not decode image: {error}"))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let expected = info.width as usize * info.height as usize * 3;
    if output.stdout.len() < expected {
        return Err(format!(
            "short image: got {} bytes, expected {expected}",
            output.stdout.len()
        ));
    }
    let mut rgb = output.stdout;
    rgb.truncate(expected);
    Ok(Frame {
        width: info.width,
        height: info.height,
        rgb,
    })
}

fn decode_alpha(info: &ImageInfo, width: u32, height: u32) -> Result<Option<Vec<u8>>, String> {
    if !info.has_alpha {
        return Ok(None);
    }
    let filter =
        format!("format=rgba,alphaextract,format=gray,scale={width}:{height}:flags=lanczos");
    let output = quiet(&mut Command::new(crate::media::ffmpeg()))
        .args(["-v", "error", "-i"])
        .arg(&info.path)
        .args([
            "-frames:v",
            "1",
            "-vf",
            &filter,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ])
        .output()
        .map_err(|error| format!("could not decode image transparency: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let expected = width as usize * height as usize;
    if output.stdout.len() < expected {
        return Err(format!(
            "short alpha plane: got {} bytes, expected {expected}",
            output.stdout.len()
        ));
    }
    let mut alpha = output.stdout;
    alpha.truncate(expected);
    Ok(Some(alpha))
}

fn write_png(frame: &Frame, alpha: Option<&[u8]>, path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    let pixel_format = if alpha.is_some() { "rgba" } else { "rgb24" };
    let pixels = if let Some(alpha) = alpha {
        if alpha.len() != frame.width as usize * frame.height as usize {
            return Err("alpha plane does not match image dimensions".into());
        }
        let mut rgba = Vec::with_capacity(alpha.len() * 4);
        for (rgb, alpha) in frame.rgb.chunks_exact(3).zip(alpha.iter().copied()) {
            rgba.extend_from_slice(rgb);
            rgba.push(alpha);
        }
        rgba
    } else {
        frame.rgb.clone()
    };

    let mut child = quiet(&mut Command::new(crate::media::ffmpeg()))
        .args([
            "-v",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-video_size",
            &format!("{}x{}", frame.width, frame.height),
            "-i",
            "-",
            "-frames:v",
            "1",
            "-c:v",
            "png",
            "-compression_level",
            "6",
        ])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start PNG encoder: {error}"))?;

    let write_result = child
        .stdin
        .take()
        .ok_or_else(|| "PNG encoder has no input pipe".to_string())?
        .write_all(&pixels);
    if let Err(error) = write_result {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_file(path);
        return Err(format!("writing PNG pixels: {error}"));
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("waiting for PNG encoder: {error}"))?;
    if !output.status.success() {
        let _ = std::fs::remove_file(path);
        return Err(format!(
            "PNG encoder failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

/// Writes a finished RGB image while restoring the source alpha plane at the
/// final dimensions. Generative refinement deliberately remains RGB-only.
pub fn write_output(info: &ImageInfo, frame: &Frame, path: &Path) -> Result<(), String> {
    let alpha = decode_alpha(info, frame.width, frame.height)?;
    write_png(frame, alpha.as_deref(), path)
}

pub fn finish(spec: &ExportSpec, model_output: Frame, source: &Frame) -> Frame {
    let colored = if spec.match_color {
        upscale::match_color_to_source(&model_output, source)
    } else {
        model_output
    };
    let limited = upscale::clamp_deviation(&colored, source, spec.detail_limit * 255.0);
    let blended = upscale::blend_toward_source(&limited, source, spec.strength);
    let (width, height) = spec.output_size();
    upscale::resample(&blended, width, height)
}

pub fn run(
    spec: &ExportSpec,
    engine: &mut Engine,
    overlap: u32,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<(), String> {
    if engine.window() != 1 {
        return Err("multi-frame video models are not valid in Image mode".into());
    }
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }

    let source = decode(&spec.image)?;
    let mut tile_progress = |done: u32, total: u32| {
        on_progress(done as u64, total as u64);
    };
    let output = engine
        .run_batch(
            std::slice::from_ref(&source),
            spec.tile,
            overlap,
            &mut tile_progress,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| "image model returned no frame".to_string())?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let final_image = finish(spec, output, &source);
    write_output(&spec.image, &final_image, &spec.out_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/logo.png")
    }

    #[test]
    fn probes_and_decodes_still_image_without_video_geometry_rules() {
        let info = probe(&logo()).expect("probe logo");
        assert_eq!((info.width, info.height), (256, 256));
        assert!(info.has_alpha);
        let frame = decode(&info).expect("decode logo");
        assert_eq!(frame.rgb.len(), 256 * 256 * 3);
        let alpha = decode_alpha(&info, 512, 512)
            .expect("decode transparency")
            .expect("alpha plane");
        assert_eq!(alpha.len(), 512 * 512);
    }

    #[test]
    fn recognizes_supported_image_extensions() {
        assert!(is_supported(Path::new("photo.PNG")));
        assert!(is_supported(Path::new("photo.jpeg")));
        assert!(is_supported(Path::new("photo.webp")));
        assert!(!is_supported(Path::new("movie.mp4")));
    }

    #[test]
    #[ignore = "requires the bundled image model and CUDA runtime"]
    fn exports_real_image_model_on_cuda_with_transparency() {
        let _ = crate::cuda::prepare();
        let model = upscale::registry()
            .into_iter()
            .find(|model| model.name == "RealESRGAN_x2")
            .expect("find native 2x image model");
        assert!(!model.is_video() && !model.video_tuned);
        let mut engine = Engine::load(&model, true).expect("load image model");
        assert_eq!(engine.provider(), "CUDA");

        let image = probe(&logo()).expect("probe logo");
        let output =
            std::env::temp_dir().join(format!("kraken-image-mode-{}-2x.png", std::process::id()));
        let _ = std::fs::remove_file(&output);
        let spec = ExportSpec {
            image,
            model,
            target_scale: 2,
            strength: 1.0,
            detail_limit: 1.0,
            match_color: true,
            fp16: true,
            tile: 256,
            out_path: output.clone(),
        };
        run(
            &spec,
            &mut engine,
            16,
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .expect("export image");
        let result = probe(&output).expect("probe output");
        assert_eq!((result.width, result.height), (512, 512));
        assert!(result.has_alpha, "output transparency was not preserved");
        std::fs::remove_file(output).expect("remove test output");
    }
}
