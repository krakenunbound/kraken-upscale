//! Video I/O, driven by the vendored LGPL ffmpeg binaries.
//!
//! Frames move over a pipe as raw rgb24 rather than through libavcodec
//! bindings: linking the ffmpeg libraries on Windows is a real toolchain
//! fight, and the process boundary costs almost nothing once frames are
//! streaming. Encoding targets NVENC because an LGPL build carries no
//! libx264/libx265.

use std::path::PathBuf;
use std::process::Command;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Debug, Clone)]
pub struct ClipInfo {
    pub path: PathBuf,
    pub coded_width: u32,
    pub coded_height: u32,
    /// Square-pixel processing dimensions after applying sample aspect ratio.
    pub width: u32,
    pub height: u32,
    pub sample_aspect_ratio: String,
    /// Exact CFR used at the raw-frame boundary, e.g. `24000/1001`.
    pub fps_rate: String,
    pub fps: f64,
    pub frames: u64,
    pub duration: f64,
    pub codec: String,
    pub pix_fmt: String,
    /// `progressive`, `tt`, `bb`, `tb`, `bt`, or `unknown`.
    pub field_order: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SourceTreatment {
    pub content: i32,
    pub restore: i32,
    pub deinterlace: bool,
}

impl SourceTreatment {
    /// Builds the decode filter chain.
    ///
    /// Deinterlace is gated on the broadcast profiles because it is automatic —
    /// the user asks for "TV, shot on video" and the correct handling follows.
    /// Restore deliberately is *not* gated: it is an explicit selection, and
    /// silently discarding it because of an unrelated dropdown would be worse
    /// than applying it. The Source help states this split.
    fn filters(self, clip: &ClipInfo) -> Vec<String> {
        let mut filters = Vec::new();
        let broadcast_profile = matches!(self.content, 2 | 3);
        if self.deinterlace && broadcast_profile && clip.is_interlaced() {
            filters.push("bwdif=mode=send_frame:parity=auto:deint=interlaced".into());
        }
        match self.restore {
            1 => filters.push("deblock=filter=weak:block=8".into()),
            2 => filters.push("deblock=filter=strong:block=8".into()),
            3 => {
                filters.push("deblock=filter=strong:block=8".into());
                // This FFmpeg build does not include hqdn3d. Keep the strong
                // restore path self-contained with its available, threaded
                // spatial denoiser instead.
                filters.push("bilateral=sigmaS=0.1:sigmaR=0.05:planes=7".into());
            }
            _ => {}
        }
        filters
    }
}

impl ClipInfo {
    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// "2560 × 1440 · 24 fps · 00:05"
    pub fn specs(&self) -> String {
        let secs = self.duration.round() as u64;
        format!(
            "{} × {} · {:.0} fps · {:02}:{:02}",
            self.width,
            self.height,
            self.fps,
            secs / 60,
            secs % 60
        )
    }

    /// The raw pipe has no timestamps or pixel-aspect metadata. Normalize both
    /// explicitly so preview and export see identical square-pixel CFR frames.
    pub fn decode_filter(&self, treatment: SourceTreatment) -> String {
        let mut filters = treatment.filters(self);
        let geometry = format!(
            "scale={}:{}:flags=lanczos,setsar=1,fps={}",
            self.width, self.height, self.fps_rate
        );
        filters.push(geometry);
        filters.join(",")
    }

    pub fn is_interlaced(&self) -> bool {
        matches!(self.field_order.as_str(), "tt" | "bb" | "tb" | "bt")
    }
}

fn bin_dir() -> PathBuf {
    // Prefer binaries shipped beside the executable, fall back to the vendored
    // tree so `cargo run` works from the source directory.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join("ffmpeg");
            if beside.join("ffmpeg.exe").exists() {
                return beside;
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/ffmpeg/bin")
}

pub fn ffmpeg() -> PathBuf {
    bin_dir().join("ffmpeg.exe")
}

pub fn ffprobe() -> PathBuf {
    bin_dir().join("ffprobe.exe")
}

fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

/// Reads stream metadata for a clip.
pub fn probe(path: &std::path::Path) -> Result<ClipInfo, String> {
    let out = quiet(&mut Command::new(ffprobe()))
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt,field_order,sample_aspect_ratio,display_aspect_ratio",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("could not run ffprobe: {e}"))?;

    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let field = |key: &str| -> Option<String> {
        text.lines()
            .find_map(|l| l.split_once('=').filter(|(k, _)| *k == key))
            .map(|(_, v)| v.trim().to_string())
    };

    let coded_width: u32 = field("width").and_then(|v| v.parse().ok()).unwrap_or(0);
    let coded_height: u32 = field("height").and_then(|v| v.parse().ok()).unwrap_or(0);
    if coded_width == 0 || coded_height == 0 {
        return Err("no video stream found".into());
    }

    let rational = |v: &str| -> Option<(u64, u64)> {
        let (n, d) = v.split_once('/')?;
        let n = n.parse::<u64>().ok()?;
        let d = d.parse::<u64>().ok()?;
        (n > 0 && d > 0).then_some((n, d))
    };

    // Prefer the average rate for a CFR representation. `r_frame_rate` can be
    // a codec timebase or the highest rate observed in VFR material.
    let (fps_num, fps_den) = field("avg_frame_rate")
        .as_deref()
        .and_then(rational)
        .or_else(|| field("r_frame_rate").as_deref().and_then(rational))
        .unwrap_or((24, 1));
    let fps = fps_num as f64 / fps_den as f64;
    let fps_rate = format!("{fps_num}/{fps_den}");

    let reported_sar = field("sample_aspect_ratio").unwrap_or_else(|| "1:1".into());
    let parsed_sar = reported_sar
        .split_once(':')
        .and_then(|(n, d)| Some((n.parse::<f64>().ok()?, d.parse::<f64>().ok()?)))
        .filter(|(n, d)| *n > 0.0 && *d > 0.0);
    let (sample_aspect_ratio, sar) = parsed_sar
        .map(|(n, d)| (reported_sar, n / d))
        .unwrap_or_else(|| ("1:1".into(), 1.0));
    // Use an even square-pixel width so yuv420p/NVENC always has valid geometry.
    let mut width = (coded_width as f64 * sar).round().max(2.0) as u32;
    if !width.is_multiple_of(2) {
        width = width.saturating_add(1);
    }
    let height = if coded_height.is_multiple_of(2) {
        coded_height
    } else {
        coded_height.saturating_add(1)
    };

    let duration = field("duration")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);

    // nb_frames is absent or "N/A" in plenty of containers; derive it instead.
    // The decoder applies an explicit fps filter, so progress must describe
    // that normalized output rather than the source packet/frame count.
    let frames = if duration > 0.0 {
        (duration * fps).round().max(0.0) as u64
    } else {
        field("nb_frames")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    };

    Ok(ClipInfo {
        path: path.to_path_buf(),
        coded_width,
        coded_height,
        width,
        height,
        sample_aspect_ratio,
        fps_rate,
        fps,
        frames,
        duration,
        codec: field("codec_name").unwrap_or_else(|| "unknown".into()),
        pix_fmt: field("pix_fmt").unwrap_or_else(|| "unknown".into()),
        field_order: field("field_order").unwrap_or_else(|| "unknown".into()),
    })
}

/// A decoded frame as tightly packed rgb24.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// Decodes a single frame at `at_secs`, optionally rescaled.
///
/// `scale_to` drives the preview's "after" pane with a Lanczos baseline until
/// model inference is wired in — which is the bar the model has to clear.
/// Default-treatment convenience form. The application always supplies a
/// treatment; this is what tests and measurement use.
#[allow(dead_code)]
pub fn frame_at(
    clip: &ClipInfo,
    at_secs: f64,
    scale_to: Option<(u32, u32)>,
) -> Result<Frame, String> {
    frame_at_with(clip, at_secs, scale_to, SourceTreatment::default())
}

pub fn frame_at_with(
    clip: &ClipInfo,
    at_secs: f64,
    scale_to: Option<(u32, u32)>,
    treatment: SourceTreatment,
) -> Result<Frame, String> {
    let (w, h) = scale_to.unwrap_or((clip.width, clip.height));

    let mut cmd = Command::new(ffmpeg());
    // Microsecond precision, not milliseconds. A frame at 24fps spans 41.667ms,
    // so frame 2 starts at 0.083333s — a request rounded to "0.083" lands back
    // inside frame 1 and silently returns the previous frame. Same at 0.208 for
    // frame 5. Scrubbing then shows stale frames and any frame-pair measurement
    // compares a frame against itself.
    quiet(&mut cmd).args(["-v", "error", "-ss", &format!("{at_secs:.6}")]);
    cmd.arg("-i").arg(&clip.path).args(["-frames:v", "1"]);

    let mut filters = treatment.filters(clip);
    let geometry = format!("scale={w}:{h}:flags=lanczos,setsar=1");
    filters.push(geometry);
    let filter = filters.join(",");
    cmd.args(["-vf", &filter]);

    let out = cmd
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .map_err(|e| format!("could not run ffmpeg: {e}"))?;

    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }

    let expected = (w as usize) * (h as usize) * 3;
    if out.stdout.len() < expected {
        return Err(format!(
            "short frame: got {} bytes, expected {expected}",
            out.stdout.len()
        ));
    }

    let mut rgb = out.stdout;
    rgb.truncate(expected);
    Ok(Frame {
        width: w,
        height: h,
        rgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_and_decode_normalize_sample_aspect_ratio() {
        let path = std::env::temp_dir().join(format!(
            "kraken-upscale-sar-test-{}.mkv",
            std::process::id()
        ));
        let status = quiet(&mut Command::new(ffmpeg()))
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=size=720x480:rate=30000/1001:color=black",
                "-t",
                "0.2",
                "-vf",
                "setsar=8/9",
                "-c:v",
                "ffv1",
            ])
            .arg(&path)
            .status()
            .expect("create test clip");
        assert!(status.success());

        let info = probe(&path).expect("probe test clip");
        assert_eq!((info.coded_width, info.coded_height), (720, 480));
        assert_eq!((info.width, info.height), (640, 480));
        assert_eq!(info.sample_aspect_ratio, "8:9");
        assert_eq!(info.fps_rate, "30000/1001");

        let mut interlaced = info.clone();
        interlaced.field_order = "tt".into();
        let treatment = SourceTreatment {
            content: 3,
            restore: 2,
            deinterlace: true,
        };
        let filter = interlaced.decode_filter(treatment);
        assert!(filter.contains("bwdif=mode=send_frame"));
        assert!(filter.contains("deblock=filter=strong"));
        assert!(filter.ends_with("setsar=1,fps=30000/1001"));

        let frame = frame_at(&info, 0.0, None).expect("decode square-pixel frame");
        assert_eq!((frame.width, frame.height), (640, 480));
        assert_eq!(frame.rgb.len(), 640 * 480 * 3);

        // Every user-selectable restore value must be accepted by the exact
        // vendored FFmpeg build; checking only one value previously let the
        // strongest option regress unnoticed.
        for restore in 0..=3 {
            let restored = frame_at_with(
                &info,
                0.0,
                None,
                SourceTreatment {
                    restore,
                    ..treatment
                },
            )
            .unwrap_or_else(|error| panic!("restore option {restore} failed: {error}"));
            assert_eq!((restored.width, restored.height), (640, 480));
            assert_eq!(restored.rgb.len(), 640 * 480 * 3);
        }

        let _ = std::fs::remove_file(path);
    }
}
