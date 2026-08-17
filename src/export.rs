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
use std::time::{Duration, Instant};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

use crate::interpolate::{self, Rife};
use crate::media::{self, ClipInfo, Frame};
use crate::stages::Stages;
use crate::temporal;
#[cfg(test)]
use crate::upscale::Upscaler;
use crate::upscale::{self, Engine};

#[derive(Clone)]
pub struct ExportSpec {
    pub clip: ClipInfo,
    pub model: upscale::ModelEntry,
    pub target_scale: u32,
    pub strength: f32,
    pub detail_limit: f32,
    pub source_treatment: media::SourceTreatment,
    pub match_color: bool,
    /// Temporal lock: hold invented detail still across frames.
    pub stability: bool,
    /// How strongly to hold it, before the per-region confidence gate.
    pub lock_strength: f32,
    /// Scene-cut sensitivity, 0..1. Higher cuts more readily.
    pub cut_detect: f32,
    pub interpolation_fps: Option<u32>,
    pub fp16: bool,
    pub tile: u32,
    pub out_path: PathBuf,
    /// `h264_nvenc` or `hevc_nvenc`. The vendored ffmpeg is LGPL, so there is
    /// no libx264/libx265 to fall back on.
    pub codec: String,
    pub cq: u32,
    pub copy_audio: bool,
}

impl ExportSpec {
    pub fn output_size(&self) -> (u32, u32) {
        crate::resolution::dimensions(self.clip.width, self.clip.height, self.target_scale, true)
    }

    /// Source rate after the front-end. Inverse telecine returns 30 to 24, so
    /// everything downstream has to work from this rather than `clip.fps`.
    pub fn source_fps(&self) -> f64 {
        self.source_treatment.effective_fps(&self.clip)
    }

    fn inverse_telecined(&self) -> bool {
        self.source_treatment.field_policy(&self.clip) == media::FieldPolicy::InverseTelecine
    }

    pub fn output_fps(&self) -> u32 {
        self.interpolation_fps
            .filter(|fps| *fps as f64 > self.source_fps())
            .unwrap_or_else(|| self.source_fps().round().max(1.0) as u32)
    }

    pub fn output_fps_rate(&self) -> String {
        match self
            .interpolation_fps
            .filter(|fps| *fps as f64 > self.source_fps())
        {
            Some(fps) => format!("{fps}/1"),
            // Keep the exact rational where possible: 24000/1001 is not 24,
            // and rounding it desynchronises audio over a full episode.
            None if self.inverse_telecined() => match self.clip.fps_rate.as_str() {
                "30000/1001" => "24000/1001".into(),
                "30/1" => "24/1".into(),
                _ => format!("{:.6}", self.source_fps()),
            },
            None if !self.clip.fps_rate.is_empty() => self.clip.fps_rate.clone(),
            None => "24/1".into(),
        }
    }

    pub fn output_duration(&self) -> f64 {
        let rate = self.output_fps_rate();
        let fps = rate
            .split_once('/')
            .and_then(|(numerator, denominator)| {
                let numerator = numerator.parse::<f64>().ok()?;
                let denominator = denominator.parse::<f64>().ok()?;
                (numerator > 0.0 && denominator > 0.0).then_some(numerator / denominator)
            })
            .unwrap_or_else(|| self.output_fps().max(1) as f64);
        self.output_frames() as f64 / fps
    }

    pub fn output_frames(&self) -> u64 {
        // Inverse telecine drops one frame in five before anything else runs.
        let decoded = if self.inverse_telecined() {
            (self.clip.frames as f64 * 4.0 / 5.0).round().max(1.0) as u64
        } else {
            self.clip.frames
        };

        match self
            .interpolation_fps
            .filter(|fps| *fps as f64 > self.source_fps())
        {
            Some(fps) if self.clip.duration > 0.0 => {
                (self.clip.duration * fps as f64).round().max(1.0) as u64
            }
            // No trusted duration — some MKV/TS containers and raw pipes report
            // none. Scale the source frame count by the rate ratio instead;
            // falling back to the source count would export far too few frames
            // and `-shortest` would then truncate the copied audio to match.
            Some(fps) if self.source_fps() > 0.0 => {
                let ratio = fps as f64 / self.source_fps();
                ((decoded.max(1) as f64) * ratio).round().max(1.0) as u64
            }
            _ => decoded.max(1),
        }
    }

    /// Known output-frame count for mux bounding. `None` when the container
    /// reported neither duration nor a frame count — `-t` would then be one
    /// guessed frame and would close the encoder on a real clip.
    pub fn frame_budget(&self) -> Option<u64> {
        if self.clip.duration > 0.0 || self.clip.frames > 0 {
            Some(self.output_frames())
        } else {
            None
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
    if spec.copy_audio {
        // Input 1: the original, purely so audio can be copied across.
        cmd.arg("-i").arg(&spec.clip.path);
        cmd.args([
            "-map", "0:v:0",
            // `?` keeps this optional — plenty of clips have no audio stream.
            "-map", "1:a:0?", "-c:a", "copy",
        ]);
    } else {
        cmd.args(["-map", "0:v:0", "-an"]);
    }
    cmd.args([
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
    // Bound the mux only when duration is known. `-shortest` cannot be used:
    // AI clips commonly have audio that ends a fraction early, and ffmpeg
    // would then close the raw-video pipe. `-t` trims long copied audio.
    // Without a trusted duration, omit `-t` and write every decoded frame.
    if spec.frame_budget().is_some() {
        cmd.arg("-t").arg(format!("{:.6}", spec.output_duration()));
    }
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

/// Keep ffmpeg's useful tail readable in the one-line activity log. ffmpeg can
/// emit pages of setup chatter before the actual failure; the final lines are
/// normally the ones that explain why its pipe closed.
fn diagnostic_tail(text: &str, max_lines: usize) -> String {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join(" | ")
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

/// ffmpeg can hang after stdin close or kill. Do not block the export thread
/// forever on `Child::wait`.
fn wait_child(child: &mut Child, timeout: Duration) -> Result<std::process::ExitStatus, String> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Ok(None) => {
                let _ = child.kill();
                let kill_deadline = Instant::now() + Duration::from_secs(15);
                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => return Ok(status),
                        Ok(None) if Instant::now() < kill_deadline => {
                            std::thread::sleep(Duration::from_millis(50));
                        }
                        Ok(None) => {
                            return Err("ffmpeg did not exit after kill".into());
                        }
                        Err(e) => return Err(format!("ffmpeg wait failed: {e}")),
                    }
                }
            }
            Err(e) => return Err(format!("ffmpeg wait failed: {e}")),
        }
    }
}

/// Runs an export to completion. `on_progress` receives (done, total) frames.
///
/// Returns the per-stage timing breakdown so callers can log where the time
/// went — the Phase 0 baseline the temporal lock will be measured against.
pub fn run(
    spec: &ExportSpec,
    up: &mut Engine,
    mut rife: Option<&mut Rife>,
    tile: u32,
    overlap: u32,
    cancel: Arc<AtomicBool>,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<Stages, String> {
    let run_started = Instant::now();
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

    // Both children exist from here on, and `Child` does not kill on drop, so
    // no path may return without tearing them down or ffmpeg is left running.
    let teardown = |decoder: &mut Child,
                    encoder: &mut Child,
                    decoder_err: &mut Option<JoinHandle<String>>,
                    encoder_err: &mut Option<JoinHandle<String>>| {
        let _ = decoder.kill();
        let _ = encoder.kill();
        let _ = wait_child(decoder, Duration::from_secs(15));
        let _ = wait_child(encoder, Duration::from_secs(15));
        if let Some(t) = decoder_err.take() {
            let _ = t.join();
        }
        if let Some(t) = encoder_err.take() {
            let _ = t.join();
        }
    };

    let mut src_out = match decoder.stdout.take() {
        Some(pipe) => pipe,
        None => {
            teardown(
                &mut decoder,
                &mut encoder,
                &mut decoder_err,
                &mut encoder_err,
            );
            remove_partial(&spec.out_path);
            return Err("decoder produced no output pipe".into());
        }
    };
    let enc_stdin = match encoder.stdin.take() {
        Some(pipe) => pipe,
        None => {
            teardown(
                &mut decoder,
                &mut encoder,
                &mut decoder_err,
                &mut encoder_err,
            );
            remove_partial(&spec.out_path);
            return Err("encoder accepted no input pipe".into());
        }
    };
    let mut enc_in = BufWriter::with_capacity(1 << 22, enc_stdin);

    let mut buf = vec![0u8; frame_bytes];
    let mut done = 0u64;
    let mut nothing = |_: u32, _: u32| {};
    let stages = Stages::new();

    // Temporal lock state. Only the previous output and the frame it came from
    // are kept, so the export stays streaming regardless of clip length.
    let mut previous: Option<(Frame, Frame)> = None;

    let processing = (|| -> Result<(), String> {
        let window = up.window();
        let apply_lock = spec.stability && window == 1;
        let mut pending = Vec::<Frame>::with_capacity(window);
        let mut last_output: Option<Frame> = None;

        let mut write_frame = |source: &Frame, out: Frame| -> Result<(), String> {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            if let Some(limit) = spec.frame_budget() {
                if done >= limit {
                    return Ok(());
                }
            }
            let mut final_frame = stages.post(|| {
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

            // The lock runs last, on the finished frame, so what gets held is
            // exactly what was written — locking before post-processing would
            // carry forward detail that later stages then alter.
            //
            // `source` here is the *rendered* frame, which on the interpolation
            // path is often a RIFE intermediate rather than a decoded one. That
            // is deliberate and correct: output N came from rendered frame N, so
            // the motion between consecutive outputs is the motion between
            // consecutive rendered frames. Substituting the original decoded
            // pair would be wrong, because interpolated outputs sit at
            // intermediate times and would be warped by too much motion.
            // Covered by `lock_with_interpolation_does_not_regress`.
            if apply_lock {
                if let Some((prev_source, prev_output)) = previous.as_ref() {
                    final_frame = stages.lock(|| -> Result<Frame, String> {
                        let motion = temporal::analyze_with(prev_source, source, spec.cut_detect)?;
                        stages.record_confidence(motion.mean_confidence());
                        let scale = final_frame.width as f32 / source.width.max(1) as f32;
                        Ok(temporal::lock(
                            prev_output,
                            &final_frame,
                            &motion,
                            spec.lock_strength,
                            scale,
                        ))
                    })?;
                }
                // Retaining the output means cloning it, but the *source* can
                // be moved out of the previous pair when sizes match, so only
                // one full-frame copy happens per frame instead of two. At 2x
                // of 704x1248 that is ~21MB/frame saved.
                let carried_source = match previous.take() {
                    Some((mut prev_source, _)) if prev_source.rgb.len() == source.rgb.len() => {
                        prev_source.width = source.width;
                        prev_source.height = source.height;
                        prev_source.rgb.copy_from_slice(&source.rgb);
                        prev_source
                    }
                    _ => source.clone(),
                };
                previous = Some((carried_source, final_frame.clone()));
            }

            stages.encode(|| {
                enc_in
                    .write_all(&final_frame.rgb)
                    .map_err(|e| format!("encode write failed: {e}"))
            })?;
            last_output = Some(final_frame);

            done += 1;
            stages.frame_done();
            on_progress(done, total);
            Ok(())
        };

        // Frame models use a one-frame window. A genuine video model waits for
        // its full NTCHW sequence and returns all corresponding output frames
        // together. Only this bounded window is retained, never the full clip.
        {
            let mut process_frame = |source: &Frame| -> Result<(), String> {
                // Never allow recurrent features from one shot to enter the next.
                // Close and pad the current window before a hard edit, then begin a
                // fresh temporal context with the new shot.
                if window > 1
                    && pending
                        .last()
                        .is_some_and(|last| interpolate::is_scene_cut(last, source))
                {
                    let real = pending.len();
                    let last = pending
                        .last()
                        .cloned()
                        .ok_or_else(|| "empty cut window".to_string())?;
                    while pending.len() < window {
                        pending.push(last.clone());
                    }
                    let outputs =
                        stages.model(|| up.run_batch(&pending, tile, overlap, &mut nothing))?;
                    for (prior, output) in pending
                        .iter()
                        .take(real)
                        .zip(outputs.into_iter().take(real))
                    {
                        write_frame(prior, output)?;
                    }
                    pending.clear();
                }
                pending.push(source.clone());
                if pending.len() < window {
                    return Ok(());
                }
                if cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                let outputs =
                    stages.model(|| up.run_batch(&pending, tile, overlap, &mut nothing))?;
                if outputs.len() != pending.len() {
                    return Err(format!(
                        "detail model returned {} frames for a {}-frame window",
                        outputs.len(),
                        pending.len()
                    ));
                }
                for (source, output) in pending.iter().zip(outputs) {
                    write_frame(source, output)?;
                }
                pending.clear();
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
                    while spec.frame_budget().is_none_or(|limit| output_index < limit)
                        && output_index as f64 / target_fps < interval_end - 1e-9
                    {
                        let output_time = output_index as f64 / target_fps;
                        let phase = ((output_time * source_fps) - source_index as f64)
                            .clamp(0.0, 1.0) as f32;
                        if phase <= 0.0001 {
                            process_frame(&left)?;
                        } else {
                            let generated = stages.interpolate(|| -> Result<Frame, String> {
                                let mut generated = rife.run(&left, &right, phase)?;
                                if spec.match_color && !interpolate::is_scene_cut(&left, &right) {
                                    let reference =
                                        interpolate::color_reference(&left, &right, phase);
                                    generated =
                                        upscale::match_color_to_source(&generated, &reference);
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
                // last available pair. Skip when duration is unknown — there is
                // no trusted pad count.
                if spec.frame_budget().is_some() {
                    while output_index < total {
                        process_frame(&left)?;
                        output_index += 1;
                    }
                }
            } else {
                while let Some(source) = read_source()? {
                    process_frame(&source)?;
                }
            }
        }
        // The processing closure's mutable borrows are released; pad the final partial
        // video window by repeating its final frame. Padded outputs are
        // discarded, preserving the exact clip duration.
        if !pending.is_empty() {
            let real = pending.len();
            let last = pending
                .last()
                .cloned()
                .ok_or_else(|| "empty final video window".to_string())?;
            while pending.len() < window {
                pending.push(last.clone());
            }
            let outputs = stages.model(|| up.run_batch(&pending, tile, overlap, &mut nothing))?;
            for (source, output) in pending
                .iter()
                .take(real)
                .zip(outputs.into_iter().take(real))
            {
                write_frame(source, output)?;
            }
        }
        if let Some(limit) = spec.frame_budget() {
            while done < limit {
                if cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                let Some(output) = last_output.as_ref() else {
                    break;
                };
                stages.encode(|| {
                    enc_in
                        .write_all(&output.rgb)
                        .map_err(|e| format!("encode write failed: {e}"))
                })?;
                done += 1;
                stages.frame_done();
                on_progress(done, total);
            }
        }
        Ok(())
    })();

    drop(enc_in);
    if processing.is_err() {
        let _ = decoder.kill();
        let _ = encoder.kill();
    }
    let decoder_status = wait_child(&mut decoder, Duration::from_secs(30));
    let encoder_status = wait_child(&mut encoder, Duration::from_secs(180));
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
        // Cancellation is a normal user action and callers intentionally
        // match this exact value. Do not decorate it with killed-child noise.
        if e == "cancelled" {
            return Err(e);
        }

        let mut details = Vec::new();
        if e.starts_with("encode write failed") {
            let tail = diagnostic_tail(&encoder_text, 8);
            if !tail.is_empty() {
                details.push(format!("ffmpeg encoder: {tail}"));
            } else if let Ok(status) = &encoder_status {
                details.push(format!("ffmpeg encoder exited with {status}"));
            }
        } else if e.starts_with("decode") || e.starts_with("decoder") {
            let tail = diagnostic_tail(&decoder_text, 8);
            if !tail.is_empty() {
                details.push(format!("ffmpeg decoder: {tail}"));
            } else if let Ok(status) = &decoder_status {
                details.push(format!("ffmpeg decoder exited with {status}"));
            }
        }
        details.push(format!("partial {}", stages.summary(run_started.elapsed())));
        return Err(format!("{e} · {}", details.join(" · ")));
    }
    let decoder_status = decoder_status.map_err(|e| {
        remove_partial(&spec.out_path);
        format!("decoder did not exit cleanly: {e}")
    })?;
    if !decoder_status.success() {
        remove_partial(&spec.out_path);
        return Err(format!("decoder failed: {}", decoder_text.trim()));
    }
    let encoder_status = encoder_status.map_err(|e| {
        remove_partial(&spec.out_path);
        format!("encoder did not exit cleanly: {e}")
    })?;
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
            detected_fields: Default::default(),
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
                temporal_frames: 1,
                video_tuned: false,
            },
            target_scale: scale,
            strength: 1.0,
            detail_limit: 1.0,
            source_treatment: media::SourceTreatment::default(),
            match_color: true,
            stability: false,
            lock_strength: 0.75,
            cut_detect: 0.4,
            interpolation_fps: None,
            fp16: false,
            tile: 256,
            out_path: PathBuf::from("output.mp4"),
            codec: "h264_nvenc".into(),
            cq: 19,
            copy_audio: true,
        }
    }

    #[test]
    fn output_dimensions_are_even() {
        assert_eq!(spec(641, 479, 3).output_size(), (1924, 1438));
        assert_eq!(spec(640, 480, 2).output_size(), (1280, 960));
    }

    #[test]
    fn output_duration_tracks_generated_frames_not_audio() {
        let mut job = spec(640, 480, 2);
        job.clip.duration = 17.333_333;
        job.clip.frames = 520;
        job.clip.fps = 30.0;
        job.clip.fps_rate = "30/1".into();
        job.interpolation_fps = Some(60);
        assert_eq!(job.output_frames(), 1040);
        assert!((job.output_duration() - 17.333_333).abs() < 0.000_001);
        assert_eq!(job.frame_budget(), Some(1040));
    }

    #[test]
    fn unknown_duration_has_no_encoder_time_limit() {
        let mut job = spec(640, 480, 2);
        job.clip.duration = 0.0;
        job.clip.frames = 0;
        assert_eq!(job.frame_budget(), None);
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

    #[test]
    fn diagnostic_tail_keeps_the_last_nonempty_lines() {
        let text = "setup\n\nfirst error\nsecond error\nactual cause\n";
        assert_eq!(diagnostic_tail(text, 2), "second error | actual cause");
    }

    /// Settles the FP16 question, which earlier measurements could not.
    ///
    /// The first comparison folded CUDA session creation and cuDNN autotuning
    /// into the per-frame average, so fp16 appeared *slower* than fp32. Now
    /// that warm-up is tracked separately, both can be compared on steady-state
    /// model time alone.
    ///
    /// `cargo test --release fp16_versus_fp32 -- --ignored --nocapture`
    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to compare fp16 against fp32"]
    fn fp16_versus_fp32() {
        let _ = crate::cuda::prepare();
        let source = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let temp = std::env::temp_dir();
        let pid = std::process::id();
        let input = temp.join(format!("kraken-fp-in-{pid}.mkv"));

        let status = quiet(&mut Command::new(media::ffmpeg()))
            .args(["-v", "error", "-y", "-i"])
            .arg(&source)
            .args(["-t", "0.5", "-an", "-c:v", "ffv1"])
            .arg(&input)
            .status()
            .expect("create fp test clip");
        assert!(status.success());

        let clip = media::probe(&input).expect("probe clip");
        let model = upscale::registry()
            .into_iter()
            .find(|entry| entry.name.contains("UltraSharp"))
            .expect("find UltraSharp");

        let run_one = |fp16: bool, tag: &str| {
            let out_path = temp.join(format!("kraken-fp-{tag}-{pid}.mp4"));
            let _ = std::fs::remove_file(&out_path);
            let up = Upscaler::load(&model, fp16).expect("load model");
            assert_eq!(up.provider, "CUDA");
            let mut up = Engine::Frame(up);
            let spec = ExportSpec {
                clip: clip.clone(),
                model: model.clone(),
                target_scale: 2,
                strength: 1.0,
                detail_limit: 1.0,
                source_treatment: media::SourceTreatment::default(),
                match_color: true,
                stability: false,
                lock_strength: 0.0,
                cut_detect: 0.4,
                interpolation_fps: None,
                fp16,
                tile: crate::recommended_tile(),
                out_path: out_path.clone(),
                codec: "h264_nvenc".into(),
                cq: 19,
                copy_audio: true,
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
            .expect("export");
            println!("{tag}: {}", stages.summary(wall.elapsed()));
            let _ = std::fs::remove_file(&out_path);
        };

        // Run fp32 first: whichever runs first pays any residual driver
        // warm-up, so putting fp16 second is the pessimistic ordering for the
        // result being claimed.
        run_one(false, "fp32");
        run_one(true, "fp16");

        let _ = std::fs::remove_file(input);
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
        let up = Upscaler::load(&model, true).expect("load detail model");
        assert_eq!(up.provider, "CUDA", "baseline must measure the GPU path");
        let mut up = Engine::Frame(up);

        let spec = ExportSpec {
            clip,
            model,
            target_scale: 2,
            strength: 1.0,
            detail_limit: 1.0,
            source_treatment: media::SourceTreatment::default(),
            match_color: true,
            // The point of this measurement is the unlocked reference.
            stability: false,
            lock_strength: 0.0,
            cut_detect: 0.4,
            interpolation_fps: None,
            fp16: true,
            tile: crate::recommended_tile(),
            out_path: output.clone(),
            codec: "h264_nvenc".into(),
            cq: 19,
            copy_audio: true,
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
        println!("model    {} ({}x native)", spec.model.name, up.scale());
        println!("tile     {}px", spec.tile);
        println!("{}", stages.summary(wall));
        println!("=========================\n");

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    /// Phase 1 acceptance: does the lock actually reduce flicker on a real
    /// export, and at what cost?
    ///
    /// Exports the same clip twice — unlocked, then locked — and compares
    /// frame-to-frame difference across the decoded results. Flicker is
    /// measured on the *output*, because that is what the viewer sees.
    ///
    /// `cargo test --release locked_export_reduces_flicker -- --ignored --nocapture`
    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to measure the temporal lock end to end"]
    fn locked_export_reduces_flicker() {
        let _ = crate::cuda::prepare();
        let source = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let temp = std::env::temp_dir();
        let pid = std::process::id();
        let input = temp.join(format!("kraken-lock-in-{pid}.mkv"));

        let status = quiet(&mut Command::new(media::ffmpeg()))
            .args(["-v", "error", "-y", "-i"])
            .arg(&source)
            .args(["-t", "0.75", "-an", "-c:v", "ffv1"])
            .arg(&input)
            .status()
            .expect("create lock test clip");
        assert!(status.success());

        let clip = media::probe(&input).expect("probe clip");
        let model = upscale::registry()
            .into_iter()
            .find(|entry| entry.name.contains("UltraSharp"))
            .expect("find UltraSharp");

        // Mean absolute difference between consecutive decoded output frames.
        // Lower means the picture is holding still.
        let measure = |path: &std::path::Path| -> f32 {
            let out = media::probe(path).expect("probe export");
            let step = 1.0 / out.fps.max(1.0);
            let mut total = 0.0;
            let mut n = 0;
            for i in 0..(out.frames.min(16).saturating_sub(1)) {
                let a = media::frame_at(&out, i as f64 * step, None).expect("frame a");
                let b = media::frame_at(&out, (i + 1) as f64 * step, None).expect("frame b");
                total += crate::temporal::mean_abs_diff(&a, &b);
                n += 1;
            }
            total / n.max(1) as f32
        };

        let run_export = |stability: bool, strength: f32, tag: &str| -> (f32, f32) {
            let out_path = temp.join(format!("kraken-lock-{tag}-{pid}.mp4"));
            let _ = std::fs::remove_file(&out_path);
            let up = Upscaler::load(&model, true).expect("load model");
            assert_eq!(up.provider, "CUDA");
            let mut up = Engine::Frame(up);
            let spec = ExportSpec {
                clip: clip.clone(),
                model: model.clone(),
                target_scale: 2,
                strength,
                detail_limit: 1.0,
                source_treatment: media::SourceTreatment::default(),
                match_color: true,
                stability,
                lock_strength: 0.75,
                cut_detect: 0.4,
                interpolation_fps: None,
                fp16: true,
                tile: crate::recommended_tile(),
                out_path: out_path.clone(),
                codec: "h264_nvenc".into(),
                cq: 19,
                copy_audio: true,
            };
            let started = std::time::Instant::now();
            let stages = run(
                &spec,
                &mut up,
                None,
                spec.tile,
                16,
                Arc::new(AtomicBool::new(false)),
                &mut |_, _| {},
            )
            .expect("export");
            let wall = started.elapsed();
            println!("{tag}: {}", stages.summary(wall));
            let flicker = measure(&out_path);
            let _ = std::fs::remove_file(&out_path);
            (flicker, wall.as_secs_f32())
        };

        // Strength 0 blends fully back to the source, so this run carries no
        // invented detail while passing through the identical pipeline and
        // encoder. Whatever flicker it shows is the floor: the source's own
        // frame-to-frame change plus NVENC noise, none of which the lock can
        // or should remove.
        let (floor, _) = run_export(false, 0.0, "reference");
        let (unlocked, unlocked_secs) = run_export(false, 1.0, "unlocked");
        let (locked, locked_secs) = run_export(true, 1.0, "locked");

        // Deliberately unclamped. If the locked run falls *below* the floor the
        // lock is suppressing the source's own change as well as the model's
        // invented detail — over-locking, which reads as smearing on motion.
        // Clamping at zero would hide that as a flattering "100% removed".
        let added_unlocked = unlocked - floor;
        let added_locked = locked - floor;
        let added_reduction = (1.0 - added_locked / added_unlocked.max(1e-6)) * 100.0;
        let total_reduction = (1.0 - locked / unlocked.max(1e-6)) * 100.0;
        let overhead = (locked_secs / unlocked_secs.max(1e-6) - 1.0) * 100.0;

        println!("\n=== TEMPORAL LOCK ===");
        println!("floor (no invented detail)  {floor:.3}");
        println!("unlocked                    {unlocked:.3}  (model added {added_unlocked:.3})");
        println!("locked                      {locked:.3}  (model added {added_locked:.3})");
        println!("model-added flicker removed {added_reduction:+.1}%   <- acceptance bar");
        println!("total output flicker        {total_reduction:+.1}%");
        println!("wall cost                   {overhead:+.1}%");
        if added_locked < 0.0 {
            println!(
                "WARNING: locked output is {:.3} below the no-detail floor — the lock is \
                 also suppressing genuine source change, which will smear motion",
                -added_locked
            );
        }
        println!("=====================\n");

        let _ = std::fs::remove_file(input);
        assert!(
            locked < unlocked,
            "lock must not increase flicker: {unlocked:.3} -> {locked:.3}"
        );
    }

    /// B34: does running Stability together with Frame Interpolation smear or
    /// ghost, because flow is estimated on RIFE intermediates?
    ///
    /// It should not. Output N is produced from rendered frame N whether that
    /// frame was decoded or synthesized, so flow between consecutive rendered
    /// frames is the motion between consecutive outputs. This asserts the
    /// combination does not increase flicker over interpolation alone.
    ///
    /// `cargo test --release lock_with_interpolation -- --ignored --nocapture`
    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to check Stability combined with RIFE"]
    fn lock_with_interpolation_does_not_regress() {
        let _ = crate::cuda::prepare();
        let source = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let temp = std::env::temp_dir();
        let pid = std::process::id();
        let input = temp.join(format!("kraken-b34-in-{pid}.mkv"));

        let status = quiet(&mut Command::new(media::ffmpeg()))
            .args(["-v", "error", "-y", "-i"])
            .arg(&source)
            .args(["-t", "0.3", "-an", "-c:v", "ffv1"])
            .arg(&input)
            .status()
            .expect("create B34 test clip");
        assert!(status.success());

        let clip = media::probe(&input).expect("probe clip");
        let model = upscale::registry()
            .into_iter()
            .find(|entry| entry.name == "RealESRGAN_x2")
            .expect("find RealESRGAN_x2");

        let measure = |path: &std::path::Path| -> f32 {
            let out = media::probe(path).expect("probe export");
            let step = 1.0 / out.fps.max(1.0);
            let mut total = 0.0;
            let mut n = 0;
            for i in 0..(out.frames.min(20).saturating_sub(1)) {
                let a = media::frame_at(&out, i as f64 * step, None).expect("a");
                let b = media::frame_at(&out, (i + 1) as f64 * step, None).expect("b");
                total += crate::temporal::mean_abs_diff(&a, &b);
                n += 1;
            }
            total / n.max(1) as f32
        };

        let run_one = |stability: bool, tag: &str| -> f32 {
            let out_path = temp.join(format!("kraken-b34-{tag}-{pid}.mp4"));
            let _ = std::fs::remove_file(&out_path);
            let up = Upscaler::load(&model, true).expect("load model");
            let mut up = Engine::Frame(up);
            let mut rife = Rife::load().expect("load RIFE");
            let spec = ExportSpec {
                clip: clip.clone(),
                model: model.clone(),
                target_scale: 2,
                strength: 1.0,
                detail_limit: 1.0,
                source_treatment: media::SourceTreatment::default(),
                match_color: true,
                stability,
                lock_strength: 0.75,
                cut_detect: 0.4,
                interpolation_fps: Some(60),
                fp16: true,
                tile: crate::recommended_tile(),
                out_path: out_path.clone(),
                codec: "h264_nvenc".into(),
                cq: 19,
                copy_audio: true,
            };
            run(
                &spec,
                &mut up,
                Some(&mut rife),
                spec.tile,
                16,
                Arc::new(AtomicBool::new(false)),
                &mut |_, _| {},
            )
            .expect("export");
            let flicker = measure(&out_path);
            let _ = std::fs::remove_file(&out_path);
            flicker
        };

        let interp_only = run_one(false, "interp");
        let interp_locked = run_one(true, "interp-locked");
        println!(
            "\nRIFE alone {interp_only:.3} · RIFE + lock {interp_locked:.3} ({:+.1}%)\n",
            (1.0 - interp_locked / interp_only.max(1e-6)) * 100.0
        );

        let _ = std::fs::remove_file(input);
        assert!(
            interp_locked <= interp_only * 1.02,
            "lock made interpolated output worse: {interp_only:.3} -> {interp_locked:.3}"
        );
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
        let up = Upscaler::load(&model, true).expect("load detail model");
        assert_eq!(up.provider, "CUDA");
        let mut up = Engine::Frame(up);
        let mut rife = Rife::load().expect("load RIFE model");
        let spec = ExportSpec {
            clip,
            model,
            target_scale: 2,
            strength: 1.0,
            detail_limit: 1.0,
            source_treatment: media::SourceTreatment::default(),
            match_color: true,
            stability: false,
            lock_strength: 0.0,
            cut_detect: 0.4,
            interpolation_fps: Some(60),
            fp16: true,
            tile: 256,
            out_path: output.clone(),
            codec: "h264_nvenc".into(),
            cq: 23,
            copy_audio: true,
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

    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to verify genuine multi-frame video export"]
    fn exports_real_video_model_window() {
        let _ = crate::cuda::prepare();
        let source = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let temp = std::env::temp_dir();
        let input = temp.join(format!("kraken-vsr-input-{}.mkv", std::process::id()));
        let output = temp.join(format!("kraken-vsr-output-{}.mp4", std::process::id()));
        let status = quiet(&mut Command::new(media::ffmpeg()))
            .args(["-v", "error", "-y", "-i"])
            .arg(&source)
            .args([
                "-frames:v",
                "7",
                "-vf",
                "scale=176:312:flags=lanczos,fps=24",
                "-an",
                "-c:v",
                "ffv1",
            ])
            .arg(&input)
            .status()
            .expect("create short video-model test clip");
        assert!(status.success());

        let clip = media::probe(&input).expect("probe short video clip");
        let model = upscale::registry()
            .into_iter()
            .find(|entry| entry.name.contains("NanoVSR"))
            .expect("find NanoVSR model");
        let mut engine = Engine::load(&model, true).expect("load video model");
        assert_eq!(engine.provider(), "CUDA");
        assert_eq!(engine.window(), 5);
        let spec = ExportSpec {
            clip,
            model,
            target_scale: 2,
            strength: 1.0,
            detail_limit: 1.0,
            source_treatment: media::SourceTreatment::default(),
            match_color: true,
            // Deliberately set: video models must bypass the legacy lock.
            stability: true,
            lock_strength: 1.0,
            cut_detect: 0.4,
            interpolation_fps: None,
            fp16: true,
            tile: 256,
            out_path: output.clone(),
            codec: "h264_nvenc".into(),
            cq: 23,
            copy_audio: true,
        };
        run(
            &spec,
            &mut engine,
            None,
            256,
            16,
            Arc::new(AtomicBool::new(false)),
            &mut |_, _| {},
        )
        .expect("export genuine video-model clip");

        let result = media::probe(&output).expect("probe video-model output");
        assert_eq!(result.frames, spec.output_frames());
        assert_eq!((result.width, result.height), spec.output_size());

        if std::env::var_os("KRAKEN_KEEP_TEST_OUTPUT").is_some() {
            println!("kept video-model input: {}", input.display());
            println!("kept video-model output: {}", output.display());
        } else {
            let _ = std::fs::remove_file(input);
            let _ = std::fs::remove_file(output);
        }
    }
}
