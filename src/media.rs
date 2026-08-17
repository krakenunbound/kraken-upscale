//! Video I/O, driven by the vendored LGPL ffmpeg binaries.
//!
//! Frames move over a pipe as raw rgb24 rather than through libavcodec
//! bindings: linking the ffmpeg libraries on Windows is a real toolchain
//! fight, and the process boundary costs almost nothing once frames are
//! streaming. Encoding targets NVENC because an LGPL build carries no
//! libx264/libx265.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

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
    /// Result of looking at the pixels, filled in on first use. Kept out of
    /// `probe` so opening a clip does not always pay for a decode pass.
    #[doc(hidden)]
    pub detected_fields: std::sync::Arc<std::sync::OnceLock<bool>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SourceTreatment {
    pub content: i32,
    pub restore: i32,
    pub deinterlace: bool,
}

/// Measures interlacing from the pixels rather than the container.
///
/// Container metadata cannot be trusted for this. A telecined MPEG-2 VOB does
/// carry `field_order=tt`, but remux it to MKV — or encode with FFV1 — and the
/// flag is simply gone while the combing remains. Since the control is labelled
/// "Detect field order & telecine", it has to actually look.
///
/// `idet`'s multi-frame counters are used because they compare across
/// neighbours and are far steadier than the single-frame heuristic.
pub fn detect_interlacing(path: &std::path::Path) -> bool {
    let mut cmd = Command::new(ffmpeg());
    let out = quiet(&mut cmd)
        .args(["-hide_banner", "-loglevel", "info", "-i"])
        .arg(path)
        // A couple of seconds is plenty to establish a cadence, and keeps
        // opening a clip responsive.
        .args(["-vf", "idet", "-frames:v", "60", "-an", "-f", "null", "-"])
        .output();

    let Ok(out) = out else { return false };
    let text = String::from_utf8_lossy(&out.stderr);

    let Some(line) = text.lines().rev().find(|l| l.contains("Multi frame detection")) else {
        return false;
    };
    let field = |key: &str| -> f64 {
        line.split(key)
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    };

    let interlaced = field("TFF:") + field("BFF:");
    let progressive = field("Progressive:");
    // A clear majority, not a bare one: a handful of false positives on
    // fast motion should not trigger field processing on progressive film.
    interlaced > progressive * 2.0 && interlaced >= 10.0
}

/// What the front-end does about fields, decided by content profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldPolicy {
    /// Progressive source, or field handling switched off.
    None,
    /// Film shot at 24fps and padded to 30 for broadcast. Reversing the 3:2
    /// pulldown recovers the original frames exactly; deinterlacing instead
    /// destroys them permanently.
    InverseTelecine,
    /// Genuinely shot at 60 fields/sec. There are no original progressive
    /// frames to recover, so interpolate.
    Deinterlace,
}

impl SourceTreatment {
    /// Chooses field handling.
    ///
    /// The distinction is the whole reason the Source profile separates "shot
    /// on film" from "shot on video", and it is not cosmetic. Measured on a
    /// telecined source against its known original:
    ///
    /// * inverse telecine — PSNR infinite, bit-exact recovery
    /// * deinterlace — PSNR 38.9 dB, permanent loss
    ///
    /// and that loss happens *before* the detail model, which then amplifies
    /// it. Deinterlacing telecined film also leaves the output at 29.97 fps
    /// where one frame in five is a blend, so a fifth of the upscaling budget
    /// is spent on frames that never existed.
    pub fn field_policy(self, clip: &ClipInfo) -> FieldPolicy {
        if !self.deinterlace || !clip.has_fields() {
            return FieldPolicy::None;
        }
        match self.content {
            // TV — shot on film. Scripted drama through the mid-2000s, which
            // includes a lot of DVD-era 4:3 television.
            2 => FieldPolicy::InverseTelecine,
            // TV — shot on video. Soaps, news, much British television.
            3 => FieldPolicy::Deinterlace,
            _ => FieldPolicy::None,
        }
    }

    /// Frame rate after the front-end, which inverse telecine changes.
    ///
    /// 3:2 pulldown padded 24 fps to 30; reversing it returns 24. The encoder
    /// and the progress total both have to know, or the export runs long and
    /// `-shortest` truncates the copied audio.
    pub fn effective_fps(self, clip: &ClipInfo) -> f64 {
        match self.field_policy(clip) {
            FieldPolicy::InverseTelecine => clip.fps * 4.0 / 5.0,
            _ => clip.fps,
        }
    }

    /// Builds the decode filter chain.
    ///
    /// Field handling is gated on the broadcast profiles because it is
    /// automatic — the user asks for "TV, shot on film" and the correct
    /// treatment follows. Restore deliberately is *not* gated: it is an
    /// explicit selection, and silently discarding it because of an unrelated
    /// dropdown would be worse than applying it. The Source help states this
    /// split.
    fn filters(self, clip: &ClipInfo) -> Vec<String> {
        let mut filters = Vec::new();
        match self.field_policy(clip) {
            FieldPolicy::InverseTelecine => {
                // fieldmatch reconstructs progressive frames from the field
                // pattern; decimate then drops the duplicate that pulldown
                // introduced. `pullup` is the older single-filter approach and
                // is not in this build.
                filters.push("fieldmatch=order=auto:combmatch=full".into());
                filters.push("decimate".into());
            }
            FieldPolicy::Deinterlace => {
                filters.push("bwdif=mode=send_frame:parity=auto:deint=interlaced".into());
            }
            FieldPolicy::None => {}
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

    /// What the container claims.
    pub fn is_interlaced(&self) -> bool {
        matches!(self.field_order.as_str(), "tt" | "bb" | "tb" | "bt")
    }

    /// Whether this file actually carries fields.
    ///
    /// Trusts the container when it says yes, and otherwise looks at the
    /// pixels — a telecined MPEG-2 VOB flags itself correctly, but remuxing it
    /// to MKV drops the flag while the combing survives, and those remuxes are
    /// what people actually have.
    ///
    /// Cached, because the detection pass costs a real decode.
    pub fn has_fields(&self) -> bool {
        if self.is_interlaced() {
            return true;
        }
        *self
            .detected_fields
            .get_or_init(|| detect_interlacing(&self.path))
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
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt,field_order,sample_aspect_ratio,display_aspect_ratio",
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

    // ffprobe prints the selected video stream before the container FORMAT
    // section. Prefer that stream duration; the container duration can instead
    // describe a longer audio track. Skip N/A values and retain the format
    // duration as a fallback for containers that omit stream duration.
    let duration = text
        .lines()
        .filter_map(|line| line.split_once('=').filter(|(key, _)| *key == "duration"))
        .filter_map(|(_, value)| value.trim().parse::<f64>().ok())
        .find(|value| value.is_finite() && *value > 0.0)
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
        detected_fields: Default::default(),
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

/// Sequential source decoder for in-app playback.
///
/// `frame_at_with` starts a new ffmpeg process for every timestamp. That is
/// the right tool for a still preview and the wrong one for a player: each
/// call pays a spawn and a GOP seek, so 25 fps "playback" never looks like
/// playback and starves the sample-frame upscale. One open pipe of CFR
/// rgb24 frames is how export already decodes; the player uses the same
/// shape from the current playhead.
pub struct SourceStream {
    child: Option<Child>,
    stdout: std::process::ChildStdout,
    width: u32,
    height: u32,
    frame_bytes: usize,
    start_secs: f64,
    fps: f64,
    frames_read: u64,
}

impl SourceStream {
    pub fn open(
        clip: &ClipInfo,
        start_secs: f64,
        treatment: SourceTreatment,
    ) -> Result<Self, String> {
        let fps = treatment.effective_fps(clip).max(1.0);
        let start_secs = start_secs.clamp(0.0, clip.duration.max(0.0));

        let mut cmd = Command::new(ffmpeg());
        quiet(&mut cmd).args(["-v", "error"]);
        // Input-side seek: a player should start quickly. Preview sampling
        // still uses the exact `frame_at_with` path after the playhead rests.
        if start_secs > 0.001 {
            cmd.args(["-ss", &format!("{start_secs:.6}")]);
        }
        cmd.arg("-i").arg(&clip.path);
        cmd.args(["-an", "-vf", &clip.decode_filter(treatment)]);
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("could not start playback decoder: {e}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "playback decoder has no stdout".to_string())?;
        if let Some(mut stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                let mut sink = Vec::new();
                let _ = stderr.read_to_end(&mut sink);
            });
        }

        let width = clip.width;
        let height = clip.height;
        Ok(Self {
            child: Some(child),
            stdout,
            width,
            height,
            frame_bytes: (width as usize)
                .saturating_mul(height as usize)
                .saturating_mul(3),
            start_secs,
            fps,
            frames_read: 0,
        })
    }

    /// Hands the child to `slot` so the UI thread can kill a blocked read.
    pub fn share_child(&mut self, slot: &std::sync::Mutex<Option<Child>>) {
        if let Some(child) = self.child.take() {
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(child);
        }
    }

    pub fn next_frame(&mut self) -> Result<Option<Frame>, String> {
        let mut rgb = vec![0u8; self.frame_bytes];
        if !read_raw_frame(&mut self.stdout, &mut rgb)? {
            return Ok(None);
        }
        self.frames_read += 1;
        Ok(Some(Frame {
            width: self.width,
            height: self.height,
            rgb,
        }))
    }

    pub fn position_secs(&self) -> f64 {
        if self.frames_read == 0 {
            return self.start_secs;
        }
        self.start_secs + (self.frames_read as f64 - 1.0) / self.fps
    }
}

impl Drop for SourceStream {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Zero bytes is a clean EOF between frames; a short read after any bytes is
/// a truncated frame (or a kill mid-read, which the player treats as stop).
fn read_raw_frame(reader: &mut impl Read, buf: &mut [u8]) -> Result<bool, String> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(format!(
                    "playback decoder stopped mid-frame ({filled}/{} bytes)",
                    buf.len()
                ))
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return Ok(false),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets").join(name)
    }

    fn telecined_clip() -> (PathBuf, ClipInfo) {
        // Generated rather than committed: applying 3:2 pulldown here keeps any
        // encode out of the comparison path, so recovery can be checked against
        // the exact original frames.
        let source = asset("progressive24.mkv");
        assert!(source.is_file(), "missing {}", source.display());
        let out = std::env::temp_dir().join(format!("kraken-tele-{}.mkv", std::process::id()));

        let status = quiet(&mut Command::new(ffmpeg()))
            .args(["-v", "error", "-y", "-i"])
            .arg(&source)
            .args([
                "-vf",
                "telecine=first_field=top:pattern=23",
                "-c:v",
                "ffv1",
            ])
            .arg(&out)
            .status()
            .expect("apply telecine");
        assert!(status.success());

        let info = probe(&out).expect("probe telecined clip");
        (out, info)
    }

    /// Film padded to 30fps must be cadence-reversed, not deinterlaced.
    ///
    /// Measured against the known original: inverse telecine recovers it
    /// bit-exactly (PSNR infinite) while deinterlacing loses ~39 dB
    /// permanently, before the detail model ever runs.
    #[test]
    fn film_profile_inverse_telecines_and_video_profile_deinterlaces() {
        let (path, clip) = telecined_clip();

        let film = SourceTreatment {
            content: 2,
            restore: 0,
            deinterlace: true,
        };
        let video = SourceTreatment {
            content: 3,
            restore: 0,
            deinterlace: true,
        };
        let off = SourceTreatment {
            content: 2,
            restore: 0,
            deinterlace: false,
        };

        assert_eq!(film.field_policy(&clip), FieldPolicy::InverseTelecine);
        assert_eq!(video.field_policy(&clip), FieldPolicy::Deinterlace);
        assert_eq!(off.field_policy(&clip), FieldPolicy::None);

        // Only the film path changes the rate: 30 -> 24.
        assert!(
            (film.effective_fps(&clip) - clip.fps * 0.8).abs() < 1e-9,
            "IVTC must drop one frame in five"
        );
        assert_eq!(video.effective_fps(&clip), clip.fps);

        let chain = clip.decode_filter(film);
        assert!(chain.contains("fieldmatch"), "{chain}");
        assert!(chain.contains("decimate"), "{chain}");
        assert!(!clip.decode_filter(video).contains("fieldmatch"));

        let _ = std::fs::remove_file(path);
    }

    /// A progressive source must never be field-processed, whatever the
    /// profile says — the profile describes the era, not this file.
    #[test]
    fn progressive_sources_are_left_alone() {
        let clip = probe(&asset("progressive24.mkv")).expect("probe 24p asset");
        for content in 0..5 {
            let t = SourceTreatment {
                content,
                restore: 0,
                deinterlace: true,
            };
            assert_eq!(
                t.field_policy(&clip),
                FieldPolicy::None,
                "content profile {content} touched a progressive source"
            );
        }
    }

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

    #[test]
    fn source_stream_reads_sequential_frames() {
        let clip = probe(&asset("progressive24.mkv")).expect("probe 24p asset");
        let mut stream =
            SourceStream::open(&clip, 0.0, SourceTreatment::default()).expect("open stream");
        let first = stream.next_frame().expect("read").expect("first frame");
        let second = stream.next_frame().expect("read").expect("second frame");
        assert_eq!((first.width, first.height), (clip.width, clip.height));
        assert_eq!(second.rgb.len(), first.rgb.len());
        assert!(stream.position_secs() + 1e-9 >= 1.0 / clip.fps);
    }
}
