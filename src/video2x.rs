//! Video2X as a third video pipeline beside the ONNX models and RTX VSR.
//!
//! Video2X runs its ncnn-Vulkan builds of Real-ESRGAN and Real-CUGAN, plus the
//! Anime4K shaders through libplacebo. It is an external program (AGPL-3.0)
//! that lives in `video2x/` beside the executable and is only ever run as a
//! child process, never linked.
//!
//! Video2X decodes and encodes on its own, so it cannot join the raw-frame
//! pipe in `export`. An export is up to four passes instead:
//!
//! 1. RIFE at source size, when interpolation is on (done by the caller).
//! 2. A prepare pass when the source needs Kraken's decode treatment
//!    (deinterlace, inverse telecine, restore, or non-square DVD pixels),
//!    which Video2X would otherwise ignore.
//! 3. Video2X itself, at the preset's nearest native scale.
//! 4. A finish pass: exact resize to the goal, the chosen NVENC codec and CQ,
//!    and audio copied from the source.

use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

use crate::export;
use crate::media::{self, Frame};
use crate::resolution;
use crate::upscale;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Processor {
    RealEsrgan,
    RealCugan,
    Anime4K,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preset {
    pub label: &'static str,
    /// Short form used in output file names.
    pub tag: &'static str,
    pub note: &'static str,
    processor: Processor,
    /// Real-ESRGAN model name, Real-CUGAN model folder, or Anime4K shader.
    model: &'static str,
    /// Real-CUGAN denoise level. `None` is Video2X's default, -1 (conservative),
    /// which its argument parser will not accept spelled out.
    noise: Option<i32>,
    /// Native scales the shipped weights cover. Empty means the processor
    /// renders any exact size (libplacebo shaders).
    scales: &'static [u32],
}

pub const PRESETS: [Preset; 13] = [
    Preset {
        label: "Real-ESRGAN AnimeVideo v3",
        tag: "V2X-ESRGAN-AnimeV3",
        note: "Video-tuned anime model. Kraken picks its 2x, 3x or 4x weights to fit the goal.",
        processor: Processor::RealEsrgan,
        model: "realesr-animevideov3",
        noise: None,
        scales: &[2, 3, 4],
    },
    Preset {
        label: "Real-ESRGAN Plus Anime (4x)",
        tag: "V2X-ESRGAN-PlusAnime",
        note: "Sharper still-image anime model. Always runs 4x, then resizes to the goal. Slow.",
        processor: Processor::RealEsrgan,
        model: "realesrgan-plus-anime",
        noise: None,
        scales: &[4],
    },
    Preset {
        label: "Real-CUGAN SE · Conservative",
        tag: "V2X-CUGAN-SE-Cons",
        note: "Faithful line art with light cleanup. A safe first choice for 2D animation.",
        processor: Processor::RealCugan,
        model: "models-se",
        noise: None,
        scales: &[2, 3, 4],
    },
    Preset {
        label: "Real-CUGAN SE · No denoise",
        tag: "V2X-CUGAN-SE-NoDN",
        note: "Keeps grain and texture. For clean sources only.",
        processor: Processor::RealCugan,
        model: "models-se",
        noise: Some(0),
        scales: &[2, 3, 4],
    },
    Preset {
        label: "Real-CUGAN SE · Denoise 1",
        tag: "V2X-CUGAN-SE-DN1",
        note: "Mild denoise. Weights exist at 2x only; larger goals get a final resize.",
        processor: Processor::RealCugan,
        model: "models-se",
        noise: Some(1),
        scales: &[2],
    },
    Preset {
        label: "Real-CUGAN SE · Denoise 2",
        tag: "V2X-CUGAN-SE-DN2",
        note: "Medium denoise. Weights exist at 2x only; larger goals get a final resize.",
        processor: Processor::RealCugan,
        model: "models-se",
        noise: Some(2),
        scales: &[2],
    },
    Preset {
        label: "Real-CUGAN SE · Denoise 3",
        tag: "V2X-CUGAN-SE-DN3",
        note: "Strong denoise for noisy DVD, VHS or heavily compressed sources. Can flatten fine detail.",
        processor: Processor::RealCugan,
        model: "models-se",
        noise: Some(3),
        scales: &[2, 3, 4],
    },
    Preset {
        label: "Real-CUGAN Pro · Conservative",
        tag: "V2X-CUGAN-Pro-Cons",
        note: "Newer Pro weights, same faithful intent as SE Conservative. 2x and 3x.",
        processor: Processor::RealCugan,
        model: "models-pro",
        noise: None,
        scales: &[2, 3],
    },
    Preset {
        label: "Real-CUGAN Pro · No denoise",
        tag: "V2X-CUGAN-Pro-NoDN",
        note: "Pro weights without denoise. For clean sources. 2x and 3x.",
        processor: Processor::RealCugan,
        model: "models-pro",
        noise: Some(0),
        scales: &[2, 3],
    },
    Preset {
        label: "Real-CUGAN Pro · Denoise 3",
        tag: "V2X-CUGAN-Pro-DN3",
        note: "Pro weights with strong denoise. 2x and 3x.",
        processor: Processor::RealCugan,
        model: "models-pro",
        noise: Some(3),
        scales: &[2, 3],
    },
    Preset {
        label: "Anime4K v4 · A+A",
        tag: "V2X-Anime4K-AA",
        note: "Fast shader, not a GAN. Mode A suits blurry, already-HD sources. Renders the exact goal size.",
        processor: Processor::Anime4K,
        model: "anime4k-v4-a+a",
        noise: None,
        scales: &[],
    },
    Preset {
        label: "Anime4K v4 · B+B",
        tag: "V2X-Anime4K-BB",
        note: "Fast shader. Mode B suits 720p-class sources with aliasing or ringing.",
        processor: Processor::Anime4K,
        model: "anime4k-v4-b+b",
        noise: None,
        scales: &[],
    },
    Preset {
        label: "Anime4K v4 · C+A",
        tag: "V2X-Anime4K-CA",
        note: "Fast shader. Mode C suits low-resolution or noisy sources such as SD.",
        processor: Processor::Anime4K,
        model: "anime4k-v4-c+a",
        noise: None,
        scales: &[],
    },
];

pub fn preset(index: i32) -> Preset {
    PRESETS[(index.max(0) as usize).min(PRESETS.len() - 1)]
}

impl Preset {
    /// Native scale to run for a goal that needs `needed`×. The smallest
    /// covering scale wins; when none covers it, the largest runs and the
    /// finish pass resizes the remainder. `None` renders the exact size.
    pub fn native_scale(&self, needed: u32) -> Option<u32> {
        let largest = self.scales.iter().copied().max()?;
        Some(
            self.scales
                .iter()
                .copied()
                .filter(|scale| *scale >= needed)
                .min()
                .unwrap_or(largest),
        )
    }

    /// One-line plan for the scale note under the output goal.
    pub fn scale_note(&self, needed: u32) -> String {
        match self.native_scale(needed) {
            None => "Video2X renders the exact goal size".into(),
            Some(scale) if scale >= needed => {
                format!("Video2X runs {scale}x, then resizes to the exact goal")
            }
            Some(scale) => format!(
                "{} weights stop at {scale}x; the last step to the goal is a plain resize",
                self.label
            ),
        }
    }

    fn args(&self, cmd: &mut Command, scale: Option<u32>, exact: (u32, u32)) {
        match self.processor {
            Processor::RealEsrgan => {
                cmd.args(["-p", "realesrgan", "--realesrgan-model", self.model]);
            }
            Processor::RealCugan => {
                cmd.args(["-p", "realcugan", "--realcugan-model", self.model]);
                if let Some(noise) = self.noise {
                    cmd.arg("-n").arg(noise.to_string());
                }
            }
            Processor::Anime4K => {
                cmd.args(["-p", "libplacebo", "--libplacebo-shader", self.model]);
            }
        }
        match scale {
            Some(scale) => {
                cmd.arg("-s").arg(scale.to_string());
            }
            None => {
                cmd.arg("-w").arg(exact.0.to_string());
                cmd.arg("-h").arg(exact.1.to_string());
            }
        }
    }
}

pub fn runtime_dir() -> Option<PathBuf> {
    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("video2x")));
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("video2x");
    let override_dir = std::env::var_os("KRAKEN_VIDEO2X_DIR").map(PathBuf::from);
    [beside_exe, Some(workspace), override_dir]
        .into_iter()
        .flatten()
        .find(|dir| dir.join("video2x.exe").is_file() && dir.join("models").is_dir())
}

pub fn available() -> bool {
    runtime_dir().is_some()
}

fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

/// Vulkan device index of the first discrete GPU, preferring NVIDIA. Video2X
/// defaults to device 0, which on some systems is the integrated GPU.
fn device() -> u32 {
    static DEVICE: OnceLock<u32> = OnceLock::new();
    *DEVICE.get_or_init(|| {
        let Some(dir) = runtime_dir() else {
            return 0;
        };
        let Ok(output) = quiet(&mut Command::new(dir.join("video2x.exe")))
            .arg("--list-devices")
            .current_dir(&dir)
            .stdin(Stdio::null())
            .output()
        else {
            return 0;
        };
        let text = String::from_utf8_lossy(&output.stdout);
        let mut devices = Vec::<(u32, String, bool)>::new();
        for line in text.lines() {
            let line = line.trim();
            if let Some((index, name)) = line.split_once(". ") {
                if let Ok(index) = index.parse::<u32>() {
                    devices.push((index, name.to_string(), false));
                    continue;
                }
            }
            if line.starts_with("Type:") && line.contains("Discrete") {
                if let Some(last) = devices.last_mut() {
                    last.2 = true;
                }
            }
        }
        devices
            .iter()
            .find(|(_, name, discrete)| *discrete && name.contains("NVIDIA"))
            .or_else(|| devices.iter().find(|(_, _, discrete)| *discrete))
            .map(|(index, _, _)| *index)
            .unwrap_or(0)
    })
}

/// Shared between the stdout reader thread and the waiting export thread.
#[derive(Default)]
struct Progress {
    done: AtomicU64,
    total: AtomicU64,
    succeeded: AtomicBool,
    tail: Mutex<Vec<String>>,
}

/// Video2X redraws its progress bar with `\r`, so lines are split on either
/// line ending. Hardware banners from ncnn go to stderr and are ignored.
fn read_progress(pipe: impl Read, progress: &Progress) {
    let mut reader = BufReader::new(pipe);
    let mut line = Vec::<u8>::new();
    let mut byte = [0u8; 1];
    let flush = |line: &mut Vec<u8>| {
        let text = String::from_utf8_lossy(line).trim().to_string();
        line.clear();
        if text.is_empty() {
            return;
        }
        // "frame=37/143 (25.87%); fps=9.25; ..." after an ANSI clear-line.
        if let Some(at) = text.find("frame=") {
            let counts = text[at + 6..].split(|c: char| c == ' ' || c == ';').next();
            if let Some((done, total)) = counts.and_then(|value| value.split_once('/')) {
                if let (Ok(done), Ok(total)) = (done.parse::<u64>(), total.parse::<u64>()) {
                    progress.done.store(done, Ordering::Relaxed);
                    progress.total.store(total, Ordering::Relaxed);
                }
            }
            return;
        }
        if text.contains("Video processed successfully") {
            progress.succeeded.store(true, Ordering::Relaxed);
        }
        if let Ok(mut tail) = progress.tail.lock() {
            tail.push(text);
            let excess = tail.len().saturating_sub(12);
            tail.drain(..excess);
        }
    };
    while let Ok(1) = reader.read(&mut byte) {
        match byte[0] {
            b'\r' | b'\n' => flush(&mut line),
            other => line.push(other),
        }
    }
    flush(&mut line);
}

fn kill(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Waits for a child, killing it on cancel. stderr is drained so a chatty
/// process cannot fill the pipe and stall.
fn wait_cancellable(
    child: &mut Child,
    cancel: &AtomicBool,
    mut tick: impl FnMut(),
) -> Result<std::process::ExitStatus, String> {
    loop {
        if cancel.load(Ordering::Relaxed) {
            kill(child);
            return Err("cancelled".into());
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                tick();
                return Ok(status);
            }
            Ok(None) => {
                tick();
                std::thread::sleep(Duration::from_millis(150));
            }
            Err(error) => return Err(format!("wait failed: {error}")),
        }
    }
}

fn drain(pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        let mut pipe = pipe;
        let _ = pipe.read_to_string(&mut text);
        text
    })
}

fn tail_lines(text: &str, count: usize) -> String {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    lines[lines.len().saturating_sub(count)..].join(" | ")
}

/// Runs ffmpeg to completion with cancel support.
fn run_ffmpeg(cmd: &mut Command, cancel: &AtomicBool, what: &str) -> Result<(), String> {
    let mut child = quiet(cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start {what}: {error}"))?;
    let stderr = child.stderr.take().map(drain);
    let status = wait_cancellable(&mut child, cancel, || {});
    let text = stderr
        .and_then(|thread| thread.join().ok())
        .unwrap_or_default();
    match status? {
        status if status.success() => Ok(()),
        status => Err(format!("{what} exited {status}: {}", tail_lines(&text, 6))),
    }
}

/// One Video2X run, file to file. The intermediate is HEVC at CQ 12 so the
/// finish pass starts from something close to what Video2X produced.
fn run_pass(
    input: &Path,
    output: &Path,
    preset: Preset,
    scale: Option<u32>,
    exact: (u32, u32),
    intermediate: bool,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<(), String> {
    let dir = runtime_dir().ok_or("Video2X runtime missing — restore the video2x folder")?;
    let mut cmd = Command::new(dir.join("video2x.exe"));
    quiet(&mut cmd).current_dir(&dir);
    cmd.arg("-i").arg(input).arg("-o").arg(output);
    cmd.arg("-d").arg(device().to_string());
    preset.args(&mut cmd, scale, exact);
    if intermediate {
        cmd.args(["-c", "hevc_nvenc", "-e", "cq=12", "-e", "preset=p5"]);
    } else {
        cmd.args(["-c", "png"]);
    }
    cmd.arg("--no-copy-streams");
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start Video2X: {error}"))?;

    let progress = Arc::new(Progress::default());
    let reader = child.stdout.take().map(|pipe| {
        let progress = progress.clone();
        std::thread::spawn(move || read_progress(pipe, &progress))
    });
    let stderr = child.stderr.take().map(drain);
    let mut reported = 0u64;
    let status = wait_cancellable(&mut child, cancel, || {
        let done = progress.done.load(Ordering::Relaxed);
        if done != reported {
            reported = done;
            on_progress(done, progress.total.load(Ordering::Relaxed));
        }
    });
    if let Some(thread) = reader {
        let _ = thread.join();
    }
    let stderr = stderr
        .and_then(|thread| thread.join().ok())
        .unwrap_or_default();
    if let Err(error) = status {
        let _ = std::fs::remove_file(output);
        return Err(error);
    }

    // Video2X 6.4 on Windows can crash during teardown after it has written a
    // complete file, so the exit code alone is not trusted either way.
    let written = std::fs::metadata(output)
        .map(|meta| meta.len() > 0)
        .unwrap_or(false);
    if progress.succeeded.load(Ordering::Relaxed) && written {
        return Ok(());
    }
    let _ = std::fs::remove_file(output);
    let tail = progress
        .tail
        .lock()
        .map(|lines| lines.join(" | "))
        .unwrap_or_default();
    Err(format!(
        "Video2X failed: {}",
        if tail.is_empty() {
            tail_lines(&stderr, 6)
        } else {
            tail
        }
    ))
}

fn temp_beside(out_path: &Path, label: &str) -> PathBuf {
    out_path.with_file_name(format!(
        "{}.{label}.mkv",
        out_path.file_stem().unwrap_or_default().to_string_lossy()
    ))
}

/// Full-clip export. `spec.clip` may already be the RIFE pass.
pub fn run(
    spec: &export::ExportSpec,
    preset: Preset,
    cancel: Arc<AtomicBool>,
    on_progress: &mut dyn FnMut(u64, u64),
    log: &mut dyn FnMut(String),
) -> Result<(), String> {
    let clip = &spec.clip;
    let (out_width, out_height) = spec.output_size();
    let needed = resolution::working_scale(clip.width, clip.height, spec.target_scale, true)?;
    let scale = preset.native_scale(needed);

    let prepared = temp_beside(&spec.out_path, "video2x-prep");
    let upscaled = temp_beside(&spec.out_path, "video2x-pass");
    let result = (|| -> Result<(), String> {
        let input = if clip.needs_prepare(spec.source_treatment) {
            log(format!(
                "VIDEO2X · prepare pass (field/pixel-aspect/restore treatment) → {}",
                prepared.display()
            ));
            let mut cmd = Command::new(media::ffmpeg());
            cmd.args(["-v", "error", "-y", "-i"]).arg(&clip.path);
            cmd.args(["-vf", &clip.decode_filter(spec.source_treatment)]);
            cmd.args([
                "-map", "0:v:0", "-an", "-c:v", "hevc_nvenc", "-preset", "p6", "-cq", "10",
                "-pix_fmt", "yuv420p",
            ]);
            cmd.arg(&prepared);
            run_ffmpeg(&mut cmd, &cancel, "prepare pass")?;
            prepared.clone()
        } else {
            clip.path.clone()
        };

        log(format!(
            "VIDEO2X · {} · {} → {}",
            preset.label,
            scale
                .map(|scale| format!("{scale}x native"))
                .unwrap_or_else(|| format!("exact {out_width}×{out_height}")),
            upscaled.display()
        ));
        run_pass(
            &input,
            &upscaled,
            preset,
            scale,
            (out_width, out_height),
            true,
            &cancel,
            on_progress,
        )?;

        log(format!(
            "VIDEO2X · finish pass → {out_width}×{out_height} {} CQ {} · audio {}",
            spec.codec,
            spec.cq,
            match (spec.copy_audio, spec.audio_aac) {
                (false, _) => "none",
                (true, true) => "AAC 320k",
                (true, false) => "copy",
            }
        ));
        let mut cmd = Command::new(media::ffmpeg());
        // `-n`: output paths are made unique before queueing; never replace.
        cmd.args(["-v", "error", "-n", "-i"]).arg(&upscaled);
        if spec.copy_audio {
            cmd.arg("-i").arg(&clip.path);
            cmd.args(["-map", "0:v:0", "-map", "1:a:0?"]);
            cmd.args(export::audio_codec_args(spec));
        } else {
            cmd.args(["-map", "0:v:0", "-an"]);
        }
        cmd.args([
            "-vf",
            &format!("scale={out_width}:{out_height}:flags=lanczos,setsar=1"),
            "-c:v",
            &spec.codec,
            "-cq",
            &spec.cq.to_string(),
            "-pix_fmt",
            "yuv420p",
        ]);
        if spec.out_path.extension().and_then(|ext| ext.to_str()) == Some("mp4") {
            cmd.args(["-movflags", "+faststart"]);
        }
        if spec.frame_budget().is_some() {
            cmd.arg("-t").arg(format!("{:.6}", spec.output_duration()));
        }
        cmd.arg(&spec.out_path);
        if let Err(error) = run_ffmpeg(&mut cmd, &cancel, "finish encoder") {
            let _ = std::fs::remove_file(&spec.out_path);
            return Err(error);
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&prepared);
    let _ = std::fs::remove_file(&upscaled);
    result
}

fn write_png(path: &Path, frame: &Frame) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), frame.width, frame.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    writer
        .write_image_data(&frame.rgb)
        .map_err(|error| error.to_string())
}

fn read_png(path: &Path) -> Result<Frame, String> {
    let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|error| error.to_string())?;
    let channels = info.color_type.samples();
    if channels < 3 {
        return Err(format!("unexpected {:?} PNG from Video2X", info.color_type));
    }
    let pixels = (info.width * info.height) as usize;
    let rgb = if channels == 3 {
        buf.truncate(pixels * 3);
        buf
    } else {
        buf.chunks_exact(channels)
            .take(pixels)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect()
    };
    Ok(Frame {
        width: info.width,
        height: info.height,
        rgb,
    })
}

/// Preview one frame. Video2X reads and writes single PNGs through its own
/// FFmpeg, so the preview uses the exact same weights as the export.
pub fn upscale_frame(
    source: &Frame,
    out_width: u32,
    out_height: u32,
    needed: u32,
    preset: Preset,
) -> Result<Frame, String> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let stamp = format!(
        "kraken-video2x-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let input = std::env::temp_dir().join(format!("{stamp}-in.png"));
    let output = std::env::temp_dir().join(format!("{stamp}-out.png"));
    let result = (|| {
        write_png(&input, source)?;
        run_pass(
            &input,
            &output,
            preset,
            preset.native_scale(needed),
            (out_width, out_height),
            false,
            &AtomicBool::new(false),
            &mut |_, _| {},
        )?;
        let frame = read_png(&output)?;
        Ok(if (frame.width, frame.height) == (out_width, out_height) {
            frame
        } else {
            upscale::resample(&frame, out_width, out_height)
        })
    })();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_scale_picks_smallest_cover_then_largest() {
        let esrgan = PRESETS[0];
        assert_eq!(esrgan.native_scale(2), Some(2));
        assert_eq!(esrgan.native_scale(3), Some(3));
        let pro = PRESETS[7];
        assert_eq!(pro.native_scale(4), Some(3));
        let dn1 = PRESETS[4];
        assert_eq!(dn1.native_scale(3), Some(2));
        assert_eq!(PRESETS[10].native_scale(3), None);
    }

    #[test]
    fn progress_parser_reads_frame_counts_and_success() {
        let progress = Progress::default();
        let text = "[info] start\r\x1b[Kframe=37/143 (25.87%); fps=9.25\r\x1b[Kframe=142/143 (99.30%); fps=12\r\n[info] Video processed successfully\r\n";
        read_progress(text.as_bytes(), &progress);
        assert_eq!(progress.done.load(Ordering::Relaxed), 142);
        assert_eq!(progress.total.load(Ordering::Relaxed), 143);
        assert!(progress.succeeded.load(Ordering::Relaxed));
    }
}
