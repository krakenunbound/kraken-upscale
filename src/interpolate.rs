//! Native RIFE frame interpolation through the same ONNX Runtime CUDA stack
//! used by the detail models.

use std::path::PathBuf;

use ort::ep::CUDA;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use rayon::prelude::*;

use crate::media::Frame;

pub const TARGET_FPS: [u32; 3] = [48, 60, 120];
const MODEL_FILE: &str = "RIFE_fp32_timestep.onnx";

pub fn target_fps(index: i32) -> u32 {
    TARGET_FPS[(index.max(0) as usize).min(TARGET_FPS.len() - 1)]
}

pub fn model_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join("models/interpolation").join(MODEL_FILE);
            if beside.is_file() {
                return beside;
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/interpolation")
        .join(MODEL_FILE)
}

pub fn available() -> bool {
    model_path().is_file()
}

pub struct Rife {
    session: Session,
    input_name: String,
    timestep_name: String,
    pub provider: &'static str,
}

impl Rife {
    pub fn load() -> Result<Self, String> {
        let path = model_path();
        if !path.is_file() {
            return Err(format!("RIFE model not found at {}", path.display()));
        }

        let session = Session::builder()
            .map_err(|e| format!("RIFE session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("RIFE optimization: {e}"))?
            .with_execution_providers(vec![CUDA::default().build().error_on_failure()])
            .map_err(|e| format!("RIFE CUDA registration: {e}"))?
            .commit_from_file(&path)
            .map_err(|e| format!("loading {}: {e}", path.display()))?;

        let input_name = session
            .inputs()
            .iter()
            .find(|input| input.name() == "input")
            .or_else(|| session.inputs().first())
            .map(|input| input.name().to_string())
            .ok_or_else(|| "RIFE graph has no frame input".to_string())?;
        let timestep_name = session
            .inputs()
            .iter()
            .find(|input| input.name() == "timestep")
            .map(|input| input.name().to_string())
            .ok_or_else(|| "RIFE graph has no timestep input".to_string())?;
        if session.outputs().is_empty() {
            return Err("RIFE graph has no output".into());
        }

        Ok(Self {
            session,
            input_name,
            timestep_name,
            provider: "CUDA",
        })
    }

    /// Produces the frame at `phase` between `left` (0) and `right` (1).
    pub fn run(&mut self, left: &Frame, right: &Frame, phase: f32) -> Result<Frame, String> {
        if left.width != right.width || left.height != right.height {
            return Err("RIFE input frame dimensions do not match".into());
        }
        if left.rgb.len() != right.rgb.len() {
            return Err("RIFE input frame buffers do not match".into());
        }

        let phase = phase.clamp(0.001, 0.999);
        if is_scene_cut(left, right) {
            return Ok(if phase < 0.5 {
                left.clone()
            } else {
                right.clone()
            });
        }

        let (w, h) = (left.width as usize, left.height as usize);
        let plane = w * h;
        let mut input = vec![0f32; plane * 6];
        input
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(channel, out)| {
                let (frame, source_channel) = if channel < 3 {
                    (left, channel)
                } else {
                    (right, channel - 3)
                };
                out.par_iter_mut().enumerate().for_each(|(pixel, value)| {
                    *value = frame.rgb[pixel * 3 + source_channel] as f32 / 255.0;
                });
            });

        let frames = Tensor::from_array(([1usize, 6, h, w], input.into_boxed_slice()))
            .map_err(|e| format!("building RIFE frame input: {e}"))?;
        let timestep = Tensor::from_array((Vec::<i64>::new(), vec![phase]))
            .map_err(|e| format!("building RIFE timestep: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                self.input_name.as_str() => frames,
                self.timestep_name.as_str() => timestep
            ])
            .map_err(|e| format!("RIFE inference: {e}"))?;
        let (_shape, output) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("reading RIFE output: {e}"))?;
        if output.len() != plane * 3 {
            return Err(format!(
                "RIFE returned {} values, expected {}",
                output.len(),
                plane * 3
            ));
        }

        let mut rgb = vec![0u8; plane * 3];
        rgb.par_chunks_mut(3)
            .enumerate()
            .for_each(|(pixel, channels)| {
                channels[0] = (output[pixel] * 255.0).clamp(0.0, 255.0) as u8;
                channels[1] = (output[plane + pixel] * 255.0).clamp(0.0, 255.0) as u8;
                channels[2] = (output[plane * 2 + pixel] * 255.0).clamp(0.0, 255.0) as u8;
            });

        Ok(Frame {
            width: left.width,
            height: left.height,
            rgb,
        })
    }
}

/// Default cut threshold: mean absolute difference as a fraction of full range.
pub(crate) const CUT_THRESHOLD: f32 = 0.28;

/// Maps the UI's 0..1 sensitivity onto a difference threshold.
///
/// Inverted, because a higher setting should cut *more* readily, which means a
/// *lower* difference is enough to count as an edit.
pub(crate) fn cut_threshold(sensitivity: f32) -> f32 {
    let s = sensitivity.clamp(0.0, 1.0);
    0.45 - 0.35 * s
}

/// Avoids generating a motion-morphed frame across a hard edit.
pub(crate) fn is_scene_cut(left: &Frame, right: &Frame) -> bool {
    is_scene_cut_with(left, right, CUT_THRESHOLD)
}

pub(crate) fn is_scene_cut_with(left: &Frame, right: &Frame, threshold: f32) -> bool {
    let (sum, samples) = left
        .rgb
        .par_chunks_exact(3)
        .zip(right.rgb.par_chunks_exact(3))
        .enumerate()
        .filter(|(index, _)| index % 16 == 0)
        .map(|(_, (a, b))| {
            let delta = (a[0] as f32 - b[0] as f32).abs()
                + (a[1] as f32 - b[1] as f32).abs()
                + (a[2] as f32 - b[2] as f32).abs();
            (delta, 3u64)
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    samples > 0 && sum / (samples as f32 * 255.0) > threshold
}

pub fn color_reference(left: &Frame, right: &Frame, phase: f32) -> Frame {
    let phase = phase.clamp(0.0, 1.0);
    let mut rgb = vec![0u8; left.rgb.len()];
    rgb.par_iter_mut().enumerate().for_each(|(index, value)| {
        *value = (left.rgb[index] as f32 * (1.0 - phase) + right.rgb[index] as f32 * phase)
            .clamp(0.0, 255.0) as u8;
    });
    Frame {
        width: left.width,
        height: left.height,
        rgb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_index_is_bounded() {
        assert_eq!(target_fps(-1), 48);
        assert_eq!(target_fps(1), 60);
        assert_eq!(target_fps(99), 120);
    }

    #[test]
    fn color_reference_blends_endpoints() {
        let left = Frame {
            width: 1,
            height: 1,
            rgb: vec![0, 50, 100],
        };
        let right = Frame {
            width: 1,
            height: 1,
            rgb: vec![100, 150, 200],
        };
        assert_eq!(color_reference(&left, &right, 0.5).rgb, [50, 100, 150]);
    }

    #[test]
    #[ignore = "requires the downloaded RIFE model and CUDA runtime"]
    fn runs_real_model() {
        let _ = crate::cuda::prepare();
        let left = Frame {
            width: 64,
            height: 64,
            rgb: vec![32; 64 * 64 * 3],
        };
        let right = Frame {
            width: 64,
            height: 64,
            rgb: vec![96; 64 * 64 * 3],
        };
        let mut rife = Rife::load().expect("load RIFE");
        let output = rife.run(&left, &right, 0.5).expect("run RIFE");
        assert_eq!((output.width, output.height), (64, 64));
        assert_eq!(output.rgb.len(), 64 * 64 * 3);
    }

    #[test]
    #[ignore = "set KRAKEN_TEST_CLIP to verify full-size RIFE and color matching"]
    fn runs_full_size_clip_pair_with_color_match() {
        let _ = crate::cuda::prepare();
        let path = std::env::var_os("KRAKEN_TEST_CLIP")
            .map(PathBuf::from)
            .expect("KRAKEN_TEST_CLIP is required");
        let clip = crate::media::probe(&path).expect("probe test clip");
        let left = crate::media::frame_at(&clip, 0.0, None).expect("decode left frame");
        let right =
            crate::media::frame_at(&clip, 1.0 / clip.fps, None).expect("decode right frame");
        let mut rife = Rife::load().expect("load RIFE");
        let started = std::time::Instant::now();
        let generated = rife.run(&left, &right, 0.5).expect("run full-size RIFE");
        let reference = color_reference(&left, &right, 0.5);
        let matched = crate::upscale::match_color_to_source(&generated, &reference);
        eprintln!(
            "RIFE full-size pair: {}x{} in {:.3}s",
            matched.width,
            matched.height,
            started.elapsed().as_secs_f32()
        );
        assert_eq!((matched.width, matched.height), (clip.width, clip.height));
        assert_eq!(matched.rgb.len(), (clip.width * clip.height * 3) as usize);
    }
}
