//! Native multi-frame video super-resolution.
//!
//! This is deliberately separate from `upscale.rs`: image models consume NCHW
//! and know nothing about time, while a real video model consumes NTCHW and
//! returns a sequence. Long clips are fed in bounded windows by the exporter.

use ort::ep::{CPU, CUDA};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use crate::media::Frame;
use crate::upscale::{models_dir, ModelEntry};

pub struct VideoUpscaler {
    session: Session,
    input_name: String,
    pub scale: u32,
    pub frames: usize,
    pub provider: String,
}

impl VideoUpscaler {
    pub fn load(entry: &ModelEntry) -> Result<Self, String> {
        if entry.temporal_frames <= 1 {
            return Err(format!("{} is not marked as a video model", entry.name));
        }
        let path = models_dir().join(&entry.file);
        let build = |cuda: bool| -> Result<Session, String> {
            let provider = if cuda {
                vec![CUDA::default().build().error_on_failure()]
            } else {
                vec![CPU::default().build()]
            };
            Session::builder()
                .map_err(|e| format!("video session builder: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| format!("video graph optimization: {e}"))?
                .with_execution_providers(provider)
                .map_err(|e| format!("video execution provider: {e}"))?
                .commit_from_file(&path)
                .map_err(|e| format!("loading {}: {e}", path.display()))
        };
        let (session, provider) = match build(true) {
            Ok(session) => (session, "CUDA"),
            Err(cuda) => {
                eprintln!("video CUDA unavailable, falling back to CPU: {cuda}");
                (build(false)?, "CPU")
            }
        };
        let input_name = session
            .inputs()
            .first()
            .map(|v| v.name().to_string())
            .ok_or_else(|| format!("video model {} has no input", path.display()))?;
        if session.outputs().is_empty() {
            return Err(format!("video model {} has no output", path.display()));
        }
        Ok(Self {
            session,
            input_name,
            scale: entry.scale.max(1),
            frames: entry.temporal_frames as usize,
            provider: provider.into(),
        })
    }

    /// Temporal models use more activation memory than frame models, so keep
    /// their low-resolution tile modest even on large cards.
    pub fn effective_tile(&self, requested: u32) -> u32 {
        requested.clamp(96, 320)
    }

    pub fn run(
        &mut self,
        frames: &[Frame],
        requested_tile: u32,
        overlap: u32,
        on_progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<Frame>, String> {
        if frames.len() != self.frames {
            return Err(format!(
                "video model needs {} frames, received {}",
                self.frames,
                frames.len()
            ));
        }
        let first = frames
            .first()
            .ok_or_else(|| "empty video window".to_string())?;
        if frames
            .iter()
            .any(|f| f.width != first.width || f.height != first.height)
        {
            return Err("video window contains mismatched frame sizes".into());
        }

        let tile = self.effective_tile(requested_tile);
        let overlap = overlap.min(tile / 4);
        let core = tile.saturating_sub(overlap * 2).max(32);
        let cols = first.width.div_ceil(core);
        let rows = first.height.div_ceil(core);
        let total = cols * rows;
        let (dw, dh) = (first.width * self.scale, first.height * self.scale);
        let mut output: Vec<Frame> = (0..self.frames)
            .map(|_| Frame {
                width: dw,
                height: dh,
                rgb: vec![0; dw as usize * dh as usize * 3],
            })
            .collect();

        let mut done = 0;
        on_progress(done, total);
        let mut cy = 0;
        while cy < first.height {
            let mut cx = 0;
            while cx < first.width {
                let core_w = core.min(first.width - cx);
                let core_h = core.min(first.height - cy);
                let x0 = cx.saturating_sub(overlap);
                let y0 = cy.saturating_sub(overlap);
                let x1 = (cx + core_w + overlap).min(first.width);
                let y1 = (cy + core_h + overlap).min(first.height);
                let tw = x1 - x0;
                let th = y1 - y0;
                let data = self.run_tile(frames, x0, y0, tw, th)?;

                let s = self.scale as usize;
                let out_tw = tw as usize * s;
                let out_th = th as usize * s;
                let plane = out_tw * out_th;
                let src_x = (cx - x0) as usize * s;
                let src_y = (cy - y0) as usize * s;
                let copy_w = core_w as usize * s;
                let copy_h = core_h as usize * s;
                let dst_x = cx as usize * s;
                let dst_y = cy as usize * s;

                for (t, frame) in output.iter_mut().enumerate() {
                    let time_base = t * 3 * plane;
                    for y in 0..copy_h {
                        for x in 0..copy_w {
                            let src = (src_y + y) * out_tw + src_x + x;
                            let dst = ((dst_y + y) * dw as usize + dst_x + x) * 3;
                            for c in 0..3 {
                                let value = data[time_base + c * plane + src];
                                frame.rgb[dst + c] =
                                    (value * 255.0).round().clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }

                done += 1;
                on_progress(done, total);
                cx += core;
            }
            cy += core;
        }
        Ok(output)
    }

    fn run_tile(
        &mut self,
        frames: &[Frame],
        x0: u32,
        y0: u32,
        tw: u32,
        th: u32,
    ) -> Result<Vec<f32>, String> {
        let plane = tw as usize * th as usize;
        let mut input = vec![0f32; self.frames * 3 * plane];
        for (t, frame) in frames.iter().enumerate() {
            let time_base = t * 3 * plane;
            for y in 0..th as usize {
                let row = ((y0 as usize + y) * frame.width as usize + x0 as usize) * 3;
                for x in 0..tw as usize {
                    let src = row + x * 3;
                    let dst = y * tw as usize + x;
                    for c in 0..3 {
                        input[time_base + c * plane + dst] = frame.rgb[src + c] as f32 / 255.0;
                    }
                }
            }
        }
        let tensor = Tensor::from_array((
            [1usize, self.frames, 3usize, th as usize, tw as usize],
            input.into_boxed_slice(),
        ))
        .map_err(|e| format!("building video input: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| format!("video inference: {e}"))?;
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("reading video output: {e}"))?;
        let expected = self.frames * 3 * tw as usize * th as usize * (self.scale as usize).pow(2);
        if data.len() != expected {
            return Err(format!(
                "video model returned {} values; expected {expected} for {} frames at {tw}x{th}",
                data.len(),
                self.frames
            ));
        }
        Ok(data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_metadata_requires_a_window() {
        let entry = ModelEntry {
            name: "bad".into(),
            file: "bad.onnx".into(),
            fp16_file: None,
            scale: 4,
            architecture: "video".into(),
            temporal_frames: 1,
            video_tuned: false,
        };
        assert!(VideoUpscaler::load(&entry).is_err());
    }

    #[test]
    #[ignore = "requires the staged NanoVSR model and CUDA runtime"]
    fn real_video_model_runs_on_cuda() {
        let entry = crate::upscale::registry()
            .into_iter()
            .find(|m| m.name.contains("NanoVSR"))
            .expect("NanoVSR registry entry");
        let mut up = VideoUpscaler::load(&entry).expect("load NanoVSR");
        assert_eq!(up.provider, "CUDA");
        let input: Vec<Frame> = (0..up.frames)
            .map(|t| Frame {
                width: 64,
                height: 48,
                rgb: (0..64 * 48 * 3)
                    .map(|i| ((i + t as u32 * 7) % 256) as u8)
                    .collect(),
            })
            .collect();
        let output = up
            .run(&input, 128, 16, &mut |_, _| {})
            .expect("video inference");
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|f| f.width == 256 && f.height == 192));
    }
}
