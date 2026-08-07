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

use crate::media::{self, ClipInfo, Frame};
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
    let fps = if spec.clip.fps_rate.is_empty() {
        "24/1"
    } else {
        spec.clip.fps_rate.as_str()
    };

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
        fps,
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
pub fn run(
    spec: &ExportSpec,
    up: &mut Upscaler,
    tile: u32,
    overlap: u32,
    cancel: Arc<AtomicBool>,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<(), String> {
    let (sw, sh) = (spec.clip.width, spec.clip.height);
    let frame_bytes = (sw as usize) * (sh as usize) * 3;
    let total = spec.clip.frames.max(1);

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

    let processing = (|| -> Result<(), String> {
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }

            if !read_frame(&mut src_out, &mut buf)? {
                break;
            }

            let source = Frame {
                width: sw,
                height: sh,
                rgb: std::mem::replace(&mut buf, vec![0u8; frame_bytes]),
            };

            let out = up.run(&source, tile, overlap, &mut nothing)?;
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let colored = if spec.match_color {
                upscale::match_color_to_source(&out, &source)
            } else {
                out
            };
            let limited = upscale::clamp_deviation(&colored, &source, spec.detail_limit * 255.0);
            let blended = upscale::blend_toward_source(&limited, &source, spec.strength);
            let (ow, oh) = spec.output_size();
            let final_frame = upscale::resample(&blended, ow, oh);

            enc_in
                .write_all(&final_frame.rgb)
                .map_err(|e| format!("encode write failed: {e}"))?;

            done += 1;
            on_progress(done, total);
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

    Ok(())
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
}
