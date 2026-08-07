//! Full-clip export: decode → upscale → encode, streamed frame by frame.
//!
//! Nothing is written to disk between stages. ffmpeg decodes to a raw rgb24
//! pipe, each frame goes through the model, and the result is fed straight to
//! an encoder process. A 4x 1080p export would otherwise need hundreds of
//! gigabytes of intermediate PNGs.

use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

use crate::interpolate::{self, Rife};
use crate::media::{self, ClipInfo, Frame};
use crate::stages::Stages;
use crate::upscale::{self, Upscaler};

#[derive(Clone)]
pub struct ExportSpec {
    pub clip: ClipInfo,
    pub model: upscale::ModelEntry,
    pub target_scale: u32,
    pub strength: f32,
    pub detail_limit: f32,
    pub source_treatment: media::SourceTreatment,
    pub match_color: bool,
    pub interpolation_fps: Option<u32>,
    pub fp16: bool,
    pub tile: u32,
    pub out_path: PathBuf,
    /// `h264_nvenc` or `hevc_nvenc`. The vendored ffmpeg is LGPL, so there is
    /// no libx264/libx265 to fall back on.
    pub codec: String,
    pub cq: u32,
}

impl ExportSpec {
    pub fn output_size(&self) -> (u32, u32) {
        let even = |n: u32| {
            if n.is_multiple_of(2) {
                n
            } else {
                n.saturating_add(1)
            }
        };
        (
            even(self.clip.width.saturating_mul(self.target_scale)),
            even(self.clip.height.saturating_mul(self.target_scale)),
        )
    }

    pub fn output_fps(&self) -> u32 {
        self.interpolation_fps
            .filter(|fps| *fps as f64 > self.clip.fps)
            .unwrap_or_else(|| self.clip.fps.round().max(1.0) as u32)
    }

    pub fn output_fps_rate(&self) -> String {
        match self
            .interpolation_fps
            .filter(|fps| *fps as f64 > self.clip.fps)
        {
            Some(fps) => format!("{fps}/1"),
            None if !self.clip.fps_rate.is_empty() => self.clip.fps_rate.clone(),
            None => "24/1".into(),
        }
    }

    pub fn output_frames(&self) -> u64 {
        match self
            .interpolation_fps
            .filter(|fps| *fps as f64 > self.clip.fps)
        {
            Some(fps) if self.clip.duration > 0.0 => {
                (self.clip.duration * fps as f64).round().max(1.0) as u64
            }
            _ => self.clip.frames.max(1),
        }
    }
}

fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

fn spawn_decoder(clip: &ClipInfo, treatment: media::SourceTreatment) -> Result<Child, String> {
    let mut cmd = Command::new(media::ffmpeg());
    quiet(&mut cmd).args(["-v", "error", "-i"]);
    cmd.arg(&clip.path);
    cmd.args(["-vf", &clip.decode_filter(treatment)]);
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start decoder: {e}"))
}

fn spawn_encoder(spec: &ExportSpec) -> Result<Child, String> {
    let (w, h) = spec.output_size();
    let fps = spec.output_fps_rate();

    let mut cmd = Command::new(media::ffmpeg());
    // Never replace a file behind the user's back. Output paths are made
    // unique before the task is queued; `-n` closes the remaining race.
    quiet(&mut cmd).args(["-v", "error", "-n"]);
    // Input 0: our raw frames.
    cmd.args([
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        &format!("{w}x{h}"),
        "-r",
        &fps,
        "-i",
        "pipe:0",
    ]);
    // Input 1: the original, purely so audio can be copied across.
    cmd.arg("-i").arg(&spec.clip.path);
    cmd.args([
        "-map",
        "0:v:0",
        // `?` keeps this optional — plenty of clips have no audio stream.
        "-map",
        "1:a:0?",
        "-c:a",
        "copy",
        "-c:v",
        &spec.codec,
        "-cq",
        &spec.cq.to_string(),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-shortest",
    ]);
    cmd.arg(&spec.out_path);

    cmd.stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start encoder: {e}"))
}

fn drain(mut pipe: impl Read + Send + 'static) -> JoinHandle<String> {
    std::thread::spawn(move || {
        let mut text = String::new();
        let _ = pipe.read_to_string(&mut text);
        text
    })
}

/// Fills one raw frame while preserving the distinction `read_exact` loses:
/// zero bytes means clean EOF between frames; EOF after any bytes is truncation.
fn read_frame(reader: &mut impl Read, buf: &mut [u8]) -> Result<bool, String> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(format!(
                    "decoder stopped mid-frame ({filled}/{} bytes)",
                    buf.len()
                ))
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(format!("decode read failed: {e}")),
        }
    }
    Ok(true)
}

fn remove_partial(path: &std::path::Path) {
    if path.is_file() {
        let _ = std::fs::remove_file(path);
    }
}

/// Runs an export to completion. `on_progress` receives (done, total) frames.
///
/// Returns the per-stage timing breakdown so callers can log where the time
/// went — the Phase 0 baseline the temporal lock will be measured against.
pub fn run(
    spec: &ExportSpec,
    up: &mut Upscaler,
    mut rife: Option<&mut Rife>,
    tile: u32,
    overlap: u32,
    cancel: Arc<AtomicBool>,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<Stages, String> {
    let (sw, sh) = (spec.clip.width, spec.clip.height);
    let frame_bytes = (sw as usize) * (sh as usize) * 3;
    let total = spec.output_frames();

    let mut decoder = spawn_decoder(&spec.clip, spec.source_treatment)?;
    let mut decoder_err = decoder.stderr.take().map(drain);
    let mut encoder = match spawn_encoder(spec) {
        Ok(child) => child,
        Err(e) => {
            let _ = decoder.kill();
            let _ = decoder.wait();
            if let Some(thread) = decoder_err.take() {
                let _ = thread.join();
            }
            return Err(e);
        }
    };
    let mut encoder_err = encoder.stderr.take().map(drain);

    let mut src_out = decoder
        .stdout
        .take()
        .ok_or_else(|| "decoder produced no output pipe".to_string())?;
    let mut enc_in = BufWriter::with_capacity(
        1 << 22,
        encoder
            .stdin
            .take()
            .ok_or_else(|| "encoder accepted no input pipe".to_string())?,
    );

    let mut buf = vec![0u8; frame_bytes];
    let mut done = 0u64;
    let mut nothing = |_: u32, _: u32| {};
    let stages = Stages::new();

    let processing = (|| -> Result<(), String> {
        let mut process_frame = |source: &Frame| -> Result<(), String> {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let out = stages.model(|| up.run(source, tile, overlap, &mut nothing))?;
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let final_frame = stages.post(|| {
                let colored = if spec.match_color {
                    upscale::match_color_to_source(&out, source)
                } else {
                    out
                };
                let limited = upscale::clamp_deviation(&colored, source, spec.detail_limit * 255.0);
                let blended = upscale::blend_toward_source(&limited, source, spec.strength);
                let (ow, oh) = spec.output_size();
                upscale::resample(&blended, ow, oh)
            });

            stages.encode(|| {
                enc_in
                    .write_all(&final_frame.rgb)
                    .map_err(|e| format!("encode write failed: {e}"))
            })?;

            done += 1;
            stages.frame_done();
            on_progress(done, total);
            Ok(())
        };

        let mut read_source = || -> Result<Option<Frame>, String> {
            if !stages.decode(|| read_frame(&mut src_out, &mut buf))? {
                return Ok(None);
            }
            Ok(Some(Frame {
                width: sw,
                height: sh,
                rgb: std::mem::replace(&mut buf, vec![0u8; frame_bytes]),
            }))
        };

        if let Some(target_fps) = spec
            .interpolation_fps
            .filter(|fps| *fps as f64 > spec.clip.fps)
        {
            let rife = rife.as_deref_mut().ok_or_else(|| {
                "frame interpolation was enabled but RIFE was not loaded".to_string()
            })?;
            let Some(mut left) = read_source()? else {
                return Err("decoder produced no frames".into());
            };
            let source_fps = spec.clip.fps.max(1.0);
            let target_fps = target_fps as f64;
            let mut source_index = 0u64;
            let mut output_index = 0u64;

            while let Some(right) = read_source()? {
                let interval_end = (source_index + 1) as f64 / source_fps;
                while output_index < total && output_index as f64 / target_fps < interval_end - 1e-9
                {
                    let output_time = output_index as f64 / target_fps;
                    let phase =
                        ((output_time * source_fps) - source_index as f64).clamp(0.0, 1.0) as f32;
                    if phase <= 0.0001 {
                        process_frame(&left)?;
                    } else {
                        let generated = stages.interpolate(|| -> Result<Frame, String> {
                            let mut generated = rife.run(&left, &right, phase)?;
                            if spec.match_color && !interpolate::is_scene_cut(&left, &right) {
                                let reference = interpolate::color_reference(&left, &right, phase);
                                generated = upscale::match_color_to_source(&generated, &reference);
                            }
                            Ok(generated)
                        })?;
                        process_frame(&generated)?;
                    }
                    output_index += 1;
                }
                left = right;
                source_index += 1;
            }

            // Preserve the source duration without inventing motion beyond the
            // last available pair.
            while output_index < total {
                process_frame(&left)?;
                output_index += 1;
            }
        } else {
            while let Some(source) = read_source()? {
                process_frame(&source)?;
            }
        }
        Ok(())
    })();

    drop(enc_in);
    if processing.is_err() {
        let _ = decoder.kill();
        let _ = encoder.kill();
    }
    let decoder_status = decoder.wait();
    let encoder_status = encoder.wait();
    let decoder_text = decoder_err
        .take()
        .and_then(|t| t.join().ok())
        .unwrap_or_default();
    let encoder_text = encoder_err
        .take()
        .and_then(|t| t.join().ok())
        .unwrap_or_default();

    if let Err(e) = processing {
        remove_partial(&spec.out_path);
        return Err(e);
    }
    let decoder_status =
        decoder_status.map_err(|e| format!("decoder did not exit cleanly: {e}"))?;
    if !decoder_status.success() {
        remove_partial(&spec.out_path);
        return Err(format!("decoder failed: {}", decoder_text.trim()));
    }
    let encoder_status =
        encoder_status.map_err(|e| format!("encoder did not exit cleanly: {e}"))?;
    if !encoder_status.success() {
        remove_partial(&spec.out_path);
        return Err(format!("encoder failed: {}", encoder_text.trim()));
    }

    Ok(stages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(width: u32, height: u32) -> ClipInfo {
        ClipInfo {
            path: PathBuf::from("input.mp4"),
            coded_width: width,
            coded_height: height,
            width,
            height,
            sample_aspect_ratio: "1:1".into(),
            fps_rate: "24/1".into(),
            fps: 24.0,
            frames: 1,
            duration: 1.0 / 24.0,
            codec: "test".into(),
            pix_fmt: "rgb24".into(),
            field_order: "progressive".into(),
        }
    }

    fn spec(width: u32, height: u32, scale: u32) -> ExportSpec {
        ExportSpec {
            clip: clip(width, height),
            model: upscale::ModelEntry {
                name: "test".into(),
                file: "test.onnx".into(),
                fp16_file: None,
                scale: 4,
                architecture: "rrdb".into(),
            },
            target_scale: scale,
            strength: 1.0,
            detail_limit: 1.0,
            source_treatment: media::SourceTreatment::default(),
            match_color: true,
            interpolation_fps: None,
            fp16: false,
            tile: 256,
            out_path: PathBuf::from("output.mp4"),
            codec: "h264_nvenc".into(),
            cq: 19,
        }
    }

    #[test]
    fn output_dimensions_are_even() {
        assert_eq!(spec(641, 479, 3).output_size(), (1924, 1438));
        assert_eq!(spec(640, 480, 2).output_size(), (1280, 960));
    }

    #[test]
    fn frame_reader_distinguishes_clean_and_partial_eof() {
        let mut exact = std::io::Cursor::new(vec![1u8; 6]);
        let mut frame = [0u8; 6];
        assert!(read_frame(&mut exact, &mut frame).unwrap());
        assert!(!read_frame(&mut exact, &mut frame).unwrap());

        let mut partial = std::io::Cursor::new(vec![1u8; 5]);
        assert!(read_frame(&mut partial, &mut frame).is_err());
    }

    /// Phase 0 of ENHANCEMENTS.md: the unlocked reference export.
    ///
    /// Prints the per-stage breakdown so the temporal lock has something
    /// concrete to be measured against, and so a later "optimisation" that
    /// moves cost between stages rather than removing it is visible.
    ///
    /// Run with:
    /// `cargo test --release baseline_stage_timings -- --ignored --nocapture`
    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to measure the unlocked export baseline"]
    fn baseline_stage_timings() {
        let _ = crate::cuda::prepare();
        let source = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let temp = std::env::temp_dir();
        let input = temp.join(format!("kraken-baseline-in-{}.mkv", std::process::id()));
        let output = temp.join(format!("kraken-baseline-out-{}.mp4", std::process::id()));

        // Half a second at native resolution: long enough for per-frame
        // averages to mean something, short enough to rerun casually.
        let status = quiet(&mut Command::new(media::ffmpeg()))
            .args(["-v", "error", "-y", "-i"])
            .arg(&source)
            .args(["-t", "0.5", "-an", "-c:v", "ffv1"])
            .arg(&input)
            .status()
            .expect("create baseline test clip");
        assert!(status.success());

        let clip = media::probe(&input).expect("probe baseline clip");
        let model = upscale::registry()
            .into_iter()
            .find(|entry| entry.name.contains("UltraSharp"))
            .expect("find UltraSharp");
        let mut up = Upscaler::load(&model, true).expect("load detail model");
        assert_eq!(up.provider, "CUDA", "baseline must measure the GPU path");

        let spec = ExportSpec {
            clip,
            model,
            target_scale: 2,
            strength: 1.0,
            detail_limit: 1.0,
            source_treatment: media::SourceTreatment::default(),
            match_color: true,
            interpolation_fps: None,
            fp16: true,
            tile: crate::recommended_tile(),
            out_path: output.clone(),
            codec: "h264_nvenc".into(),
            cq: 19,
        };

        let wall = std::time::Instant::now();
        let stages = run(
            &spec,
            &mut up,
            None,
            spec.tile,
            16,
            Arc::new(AtomicBool::new(false)),
            &mut |_, _| {},
        )
        .expect("baseline export");
        let wall = wall.elapsed();

        println!("\n=== UNLOCKED BASELINE ===");
        println!("clip     {}x{}", spec.clip.width, spec.clip.height);
        println!("model    {} ({}x native)", spec.model.name, up.scale);
        println!("tile     {}px", spec.tile);
        println!("{}", stages.summary(wall));
        println!("=========================\n");

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to verify the real CUDA RIFE export stage"]
    fn exports_real_rife_sequence() {
        let _ = crate::cuda::prepare();
        let source = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let temp = std::env::temp_dir();
        let input = temp.join(format!("kraken-rife-input-{}.mkv", std::process::id()));
        let output = temp.join(format!("kraken-rife-output-{}.mp4", std::process::id()));
        let status = quiet(&mut Command::new(media::ffmpeg()))
            .args(["-v", "error", "-y", "-i"])
            .arg(&source)
            .args([
                "-t",
                "0.25",
                "-vf",
                "scale=176:312:flags=lanczos,fps=24",
                "-an",
                "-c:v",
                "ffv1",
            ])
            .arg(&input)
            .status()
            .expect("create short RIFE test clip");
        assert!(status.success());

        let clip = media::probe(&input).expect("probe short RIFE clip");
        let model = upscale::registry()
            .into_iter()
            .find(|entry| entry.name == "RealESRGAN_x2")
            .expect("find native 2x detail model");
        let mut up = Upscaler::load(&model, true).expect("load detail model");
        assert_eq!(up.provider, "CUDA");
        let mut rife = Rife::load().expect("load RIFE model");
        let spec = ExportSpec {
            clip,
            model,
            target_scale: 2,
            strength: 1.0,
            detail_limit: 1.0,
            source_treatment: media::SourceTreatment::default(),
            match_color: true,
            interpolation_fps: Some(60),
            fp16: true,
            tile: 256,
            out_path: output.clone(),
            codec: "h264_nvenc".into(),
            cq: 23,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        run(
            &spec,
            &mut up,
            Some(&mut rife),
            256,
            16,
            cancel,
            &mut |_, _| {},
        )
        .expect("export interpolated test clip");

        let result = media::probe(&output).expect("probe interpolated output");
        assert!((result.fps - 60.0).abs() < 0.01);
        assert_eq!(result.frames, spec.output_frames());
        assert_eq!((result.width, result.height), spec.output_size());

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }
}
