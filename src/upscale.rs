//! Model inference: ONNX Runtime with overlap-blended tiling.

use std::path::{Path, PathBuf};

use ort::ep::{ExecutionProviderDispatch, CPU, CUDA};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use rayon::prelude::*;
use serde::Deserialize;

use crate::media::Frame;
use crate::pth;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub file: String,
    /// Half-precision sibling, when the converter managed to produce one.
    #[serde(default)]
    pub fp16_file: Option<String>,
    pub scale: u32,
    #[serde(default)]
    pub architecture: String,
    /// Number of neighboring frames consumed together. One means a regular
    /// image model; values above one select the genuine video-model path.
    #[serde(default = "single_frame")]
    pub temporal_frames: u32,
}

fn single_frame() -> u32 {
    1
}

impl ModelEntry {
    pub fn is_video(&self) -> bool {
        self.temporal_frames > 1
    }
}

/// What each model is actually good for.
///
/// Generic blurb text is worse than none: picking the wrong architecture for
/// the content is the single most common cause of visible artifacts, and the
/// filename alone doesn't tell you that AnimeSharp will draw ink strokes into
/// skin.
pub fn guidance(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if n.contains("nanovsr") {
        "Genuine multi-frame video super-resolution. Uses neighboring frames together so detail follows motion instead of being reinvented frame by frame. Best on low-resolution live action; the separate Stability filter is bypassed."
    } else if n.contains("anime") {
        "Trained on line art. On photographic skin it invents dark strokes \
         and hard edges — use only for animation and 2D."
    } else if n.contains("remacri") {
        "Photoreal, gentler than UltraSharp. Best general default for live \
         action and faces."
    } else if n.contains("ultrasharp") {
        "Sharp general purpose. Strong on fabric, hair and foliage. Can \
         over-crisp skin — try Strength 0.6–0.8 on close-ups."
    } else if n.contains("siax") {
        "Photoreal with aggressive fine detail. Good on textiles and \
         foliage; will exaggerate any grain or compression noise."
    } else if n.contains("faces") {
        "Face-specialised at 8x native. Poor on anything that isn't a face."
    } else if n.contains("superscale") {
        "8x native — heavy, and only worth it for very small sources."
    } else if n.contains("realesrgan") {
        "Balanced and conservative. Least likely to invent artifacts, at \
         the cost of a softer result."
    } else if n.contains("esrgan") {
        "Original ESRGAN. Superseded by the others; useful as a baseline."
    } else {
        "No notes for this model yet. Compare it against Remacri at the \
         same Strength to judge it."
    }
}

/// Warns when the model's native scale is far above the requested one.
///
/// A 4x model asked for 2x still runs at 4x internally and is then averaged
/// down, so it costs four times the pixels for the same output. That is a
/// deliberate quality lever — the downsample suppresses invented artifacts —
/// but it is a large, silent cost, and a native-scale model is usually the
/// better trade when a matching one is installed.
pub fn scale_advice(model: &ModelEntry, target: u32, installed: &[ModelEntry]) -> Option<String> {
    if model.scale <= target || model.scale == 0 {
        return None;
    }

    let factor = (model.scale / target.max(1)).pow(2);
    let alternative = installed
        .iter()
        .find(|m| m.scale == target && m.name != model.name)
        .map(|m| format!(" — {} is native {target}x", m.name))
        .unwrap_or_default();

    Some(format!(
        "{}x model for a {target}x target: ~{factor}x the work, then averaged down{alternative}.",
        model.scale
    ))
}

/// Index of the best default model for photographic content.
pub fn preferred_default(models: &[ModelEntry]) -> usize {
    // Anything anime- or face-specific is a poor first impression on the
    // live-action and AI footage this is aimed at.
    let rank = |m: &ModelEntry| -> u8 {
        let n = m.name.to_ascii_lowercase();
        if m.is_video() {
            0
        } else if n.contains("remacri") {
            1
        } else if n.contains("ultrasharp") {
            2
        } else if n.contains("siax") {
            3
        } else if n.contains("realesrgan") {
            4
        } else if n.contains("anime") || n.contains("faces") {
            10
        } else {
            6
        }
    };
    models
        .iter()
        .enumerate()
        .min_by_key(|(_, m)| rank(m))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn models_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join("models");
            if beside.is_dir() {
                return beside;
            }
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

/// Discovers available models.
///
/// The models directory is the source of truth, not the index: dropping any
/// `.onnx` in makes it selectable, with no conversion step and no rebuild.
/// `models.json`, when present, only supplies metadata that would otherwise
/// have to be probed — chiefly the native scale factor.
pub fn registry() -> Vec<ModelEntry> {
    let dir = models_dir();

    let mut meta: Vec<ModelEntry> = std::fs::read_to_string(dir.join("models.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();

    if let Ok(text) = std::fs::read_to_string(dir.join("video").join("models.json")) {
        if let Ok(mut video_meta) = serde_json::from_str::<Vec<ModelEntry>>(&text) {
            meta.append(&mut video_meta);
        }
    }

    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("no models directory at {}", dir.display());
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let extension = path.extension().and_then(|e| e.to_str());
        if !matches!(extension, Some("onnx" | "pth")) {
            continue;
        }
        let file = path.file_name().unwrap_or_default().to_string_lossy();
        // Half-precision variants are siblings, not separate models.
        if file.ends_with(".fp16.onnx") {
            continue;
        }
        let name = file
            .trim_end_matches(".onnx")
            .trim_end_matches(".pth")
            .to_string();
        // Prefer a native .pth when both forms exist; it exercises the exact
        // drop-in path while keeping one model entry in the UI.
        if seen.contains(&name) && extension == Some("onnx") {
            continue;
        }
        if extension == Some("pth") {
            if let Some(pos) = out.iter().position(|m: &ModelEntry| m.name == name) {
                out.remove(pos);
            }
        }
        seen.insert(name.clone());

        let fp16 = dir.join(format!("{name}.fp16.onnx"));
        let known = meta.iter().find(|m| m.name == name);

        out.push(ModelEntry {
            file: file.into_owned(),
            fp16_file: (extension == Some("onnx") && fp16.is_file())
                .then(|| format!("{name}.fp16.onnx")),
            // 0 means "probe it on load" — a dropped-in model has no metadata.
            scale: known.map(|m| m.scale).unwrap_or(0),
            architecture: known.map(|m| m.architecture.clone()).unwrap_or_default(),
            temporal_frames: known.map(|m| m.temporal_frames).unwrap_or(1),
            name,
        });
    }

    // Video models live in their own folder so interpolation/restoration
    // graphs are never mistaken for detail models. Their metadata is required:
    // unlike a 4-D image graph, a 5-D temporal graph cannot be safely guessed.
    let video_dir = dir.join("video");
    if let Ok(entries) = std::fs::read_dir(&video_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("onnx") {
                continue;
            }
            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
            let known = meta.iter().find(|m| {
                Path::new(&m.file).file_name().and_then(|n| n.to_str()) == Some(file_name.as_ref())
            });
            let Some(known) = known else {
                eprintln!("ignoring video model without metadata: {}", path.display());
                continue;
            };
            if known.temporal_frames <= 1 {
                eprintln!(
                    "ignoring video model with invalid temporal window: {}",
                    path.display()
                );
                continue;
            }
            let mut model = known.clone();
            model.file = format!("video/{}", file_name);
            out.push(model);
        }
    }

    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

/// One loaded detail stage. Frame models remain supported, while video models
/// receive a real temporal window through the same export/preview workers.
pub enum Engine {
    Frame(Upscaler),
    Video(crate::video::VideoUpscaler),
}

impl Engine {
    pub fn load(entry: &ModelEntry, fp16: bool) -> Result<Self, String> {
        if entry.is_video() {
            crate::video::VideoUpscaler::load(entry).map(Self::Video)
        } else {
            Upscaler::load(entry, fp16).map(Self::Frame)
        }
    }

    pub fn scale(&self) -> u32 {
        match self {
            Self::Frame(up) => up.scale,
            Self::Video(up) => up.scale,
        }
    }

    pub fn provider(&self) -> &str {
        match self {
            Self::Frame(up) => &up.provider,
            Self::Video(up) => &up.provider,
        }
    }

    pub fn window(&self) -> usize {
        match self {
            Self::Frame(_) => 1,
            Self::Video(up) => up.frames,
        }
    }

    pub fn effective_tile(&self, requested: u32) -> u32 {
        match self {
            Self::Frame(up) => up.effective_tile(requested),
            Self::Video(up) => up.effective_tile(requested),
        }
    }

    pub fn run_batch(
        &mut self,
        frames: &[Frame],
        tile: u32,
        overlap: u32,
        on_progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<Frame>, String> {
        match self {
            Self::Frame(up) => frames
                .iter()
                .map(|frame| up.run(frame, tile, overlap, on_progress))
                .collect(),
            Self::Video(up) => up.run(frames, tile, overlap, on_progress),
        }
    }
}

fn build(path: &Path, providers: Vec<ExecutionProviderDispatch>) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("optimization level: {e}"))?
        .with_execution_providers(providers)
        .map_err(|e| format!("execution providers: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("loading {}: {e}", path.display()))
}

pub struct Upscaler {
    session: Session,
    input_name: String,
    pub scale: u32,
    pub provider: String,
}

impl Upscaler {
    /// Keeps the native model output tile at or below 4096px. Larger pitches
    /// exceed CUDA indexing limits for some 8x RRDB graphs.
    pub fn effective_tile(&self, requested: u32) -> u32 {
        requested.min(4096 / self.scale.max(1)).max(64)
    }

    pub fn load(entry: &ModelEntry, fp16: bool) -> Result<Self, String> {
        if Path::new(&entry.file).extension().and_then(|e| e.to_str()) == Some("pth") {
            let (session, scale, provider) = pth::load_rrdb(&models_dir().join(&entry.file))?;
            let input_name = session
                .inputs()
                .first()
                .map(|o| o.name().to_string())
                .ok_or_else(|| "imported RRDB model has no inputs".to_string())?;
            return Ok(Self {
                session,
                input_name,
                scale,
                provider,
            });
        }
        if fp16 {
            if let Some(f) = entry.fp16_file.as_ref() {
                match Self::load_path(&models_dir().join(f), entry.scale) {
                    Ok(u) => return Ok(u),
                    // A half export that this driver or provider won't run
                    // should cost speed, not the whole render.
                    Err(e) => eprintln!("fp16 unavailable for {}, using fp32: {e}", entry.name),
                }
            }
        }
        Self::load_path(&models_dir().join(&entry.file), entry.scale)
    }

    pub fn load_path(path: &Path, scale: u32) -> Result<Self, String> {
        // Ask for CUDA and make registration failure loud, so a missing CUDA
        // runtime falls back to CPU here rather than silently running at
        // one-thirtieth the speed with no explanation.
        let (session, provider) =
            match build(path, vec![CUDA::default().build().error_on_failure()]) {
                Ok(s) => (s, "CUDA"),
                Err(cuda_err) => {
                    eprintln!("CUDA unavailable, falling back to CPU: {cuda_err}");
                    (build(path, vec![CPU::default().build()])?, "CPU")
                }
            };

        let input_name = session
            .inputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| format!("model {} has no inputs", path.display()))?;
        if session.outputs().is_empty() {
            return Err(format!("model {} has no outputs", path.display()));
        }

        let mut up = Self {
            session,
            input_name,
            scale,
            provider: provider.into(),
        };

        // A model dropped straight into the folder carries no metadata, so
        // ask it directly how much it scales by.
        if up.scale == 0 {
            up.scale = up.probe_scale()?;
        }

        Ok(up)
    }

    /// Runs a tiny input to learn the model's native scale factor.
    fn probe_scale(&mut self) -> Result<u32, String> {
        const N: usize = 32;
        let probe = Frame {
            width: N as u32,
            height: N as u32,
            rgb: vec![128u8; N * N * 3],
        };
        let out = self.run_tile(&probe, 0, 0, N as u32, N as u32)?;
        let pixels = out.len() / 3;
        let side = (pixels as f64).sqrt().round() as usize;
        if side == 0 || !side.is_multiple_of(N) {
            return Err(format!(
                "could not determine scale factor (probe returned {pixels} pixels)"
            ));
        }
        Ok((side / N) as u32)
    }

    /// Upscales a whole frame by the model's native factor.
    ///
    /// Tiles overlap and are feathered together; running a frame in one shot
    /// would blow past 24 GB the moment the source gets large, and butting
    /// tiles edge-to-edge leaves visible seams because convolution padding
    /// makes tile borders unreliable.
    pub fn run(
        &mut self,
        frame: &Frame,
        tile: u32,
        overlap: u32,
        on_progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Frame, String> {
        let tile = self.effective_tile(tile);
        let (sw, sh) = (frame.width, frame.height);
        let s = self.scale;
        let (dw, dh) = (sw * s, sh * s);

        let mut accum = vec![0f32; (dw as usize) * (dh as usize) * 3];
        let mut weight = vec![0f32; (dw as usize) * (dh as usize)];

        let step = tile.saturating_sub(overlap * 2).max(16);
        let total = sw.div_ceil(step) * sh.div_ceil(step);
        let mut done = 0u32;
        on_progress(0, total);

        let mut y = 0u32;
        while y < sh {
            let mut x = 0u32;
            while x < sw {
                // Expand by the overlap, clamped to the frame.
                let x0 = x.saturating_sub(overlap);
                let y0 = y.saturating_sub(overlap);
                let x1 = (x + step + overlap).min(sw);
                let y1 = (y + step + overlap).min(sh);
                let (tw, th) = (x1 - x0, y1 - y0);

                let out = self.run_tile(frame, x0, y0, tw, th)?;

                blend_into(
                    &mut accum,
                    &mut weight,
                    dw,
                    dh,
                    &out,
                    x0 * s,
                    y0 * s,
                    tw * s,
                    th * s,
                    overlap * s,
                );

                done += 1;
                on_progress(done, total);

                x += step;
            }
            y += step;
        }

        let mut rgb = vec![0u8; (dw as usize) * (dh as usize) * 3];
        for i in 0..(dw as usize) * (dh as usize) {
            let w = weight[i].max(1e-6);
            for c in 0..3 {
                rgb[i * 3 + c] = (accum[i * 3 + c] / w * 255.0).clamp(0.0, 255.0) as u8;
            }
        }

        Ok(Frame {
            width: dw,
            height: dh,
            rgb,
        })
    }

    fn run_tile(
        &mut self,
        frame: &Frame,
        x0: u32,
        y0: u32,
        tw: u32,
        th: u32,
    ) -> Result<Vec<f32>, String> {
        // NCHW, planar, 0..1 — the convention every ESRGAN-family export uses.
        let mut input = vec![0f32; 3 * (tw as usize) * (th as usize)];
        let plane = (tw as usize) * (th as usize);
        for ty in 0..th as usize {
            let src_row = ((y0 as usize + ty) * frame.width as usize + x0 as usize) * 3;
            for tx in 0..tw as usize {
                let p = src_row + tx * 3;
                let d = ty * tw as usize + tx;
                input[d] = frame.rgb[p] as f32 / 255.0;
                input[plane + d] = frame.rgb[p + 1] as f32 / 255.0;
                input[plane * 2 + d] = frame.rgb[p + 2] as f32 / 255.0;
            }
        }

        let tensor = Tensor::from_array((
            [1usize, 3, th as usize, tw as usize],
            input.into_boxed_slice(),
        ))
        .map_err(|e| format!("building input tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| format!("inference: {e}"))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("reading output tensor: {e}"))?;

        // A mis-tagged models.json entry, a bad scale probe, or a graph that
        // is not an ESRGAN-family upscaler would otherwise be blended with the
        // wrong stride — reading past the buffer or silently corrupting the
        // tile. Fail with the numbers instead.
        let expected = 3 * (tw as usize) * (th as usize) * (self.scale as usize).pow(2);
        if data.len() != expected {
            let ratio = (data.len() as f64 / (3.0 * tw as f64 * th as f64)).sqrt();
            return Err(format!(
                "model output does not match its {}x scale: got {} values for a {tw}x{th} tile, \
                 expected {expected} (output implies about {ratio:.2}x)",
                self.scale,
                data.len()
            ));
        }

        Ok(data.to_vec())
    }
}

/// Pulls model output back toward the source at `strength` < 1.
///
/// The source is bilinearly resampled to the model's output size and mixed in.
/// This is the main handle on hallucination: a model that invents texture you
/// don't want is usually still worth 60–80% of its output.
pub fn blend_toward_source(model_out: &Frame, source: &Frame, strength: f32) -> Frame {
    let strength = strength.clamp(0.0, 1.0);
    if strength >= 0.999 {
        return Frame {
            width: model_out.width,
            height: model_out.height,
            rgb: model_out.rgb.clone(),
        };
    }

    let (dw, dh) = (model_out.width, model_out.height);
    let (sw, sh) = (source.width, source.height);
    let mut rgb = vec![0u8; (dw as usize) * (dh as usize) * 3];

    let xr = sw as f32 / dw as f32;
    let yr = sh as f32 / dh as f32;

    rgb.par_chunks_mut(dw as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = ((y as f32 + 0.5) * yr - 0.5).max(0.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(sh as usize - 1);
            let wy = fy - y0 as f32;

            for (x, pixel) in row.chunks_mut(3).enumerate() {
                let fx = ((x as f32 + 0.5) * xr - 0.5).max(0.0);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(sw as usize - 1);
                let wx = fx - x0 as f32;

                for (c, channel) in pixel.iter_mut().enumerate() {
                    let p = |px: usize, py: usize| -> f32 {
                        source.rgb[(py * sw as usize + px) * 3 + c] as f32
                    };
                    let top = p(x0, y0) * (1.0 - wx) + p(x1, y0) * wx;
                    let bot = p(x0, y1) * (1.0 - wx) + p(x1, y1) * wx;
                    let base = top * (1.0 - wy) + bot * wy;

                    let model = model_out.rgb[(y * dw as usize + x) * 3 + c] as f32;
                    *channel = (model * strength + base * (1.0 - strength)).clamp(0.0, 255.0) as u8;
                }
            }
        });

    Frame {
        width: dw,
        height: dh,
        rgb,
    }
}

/// Matches per-channel mean and contrast to the source while limiting the
/// correction so a single unusual frame cannot produce a dramatic color jump.
pub fn match_color_to_source(model_out: &Frame, source: &Frame) -> Frame {
    fn stats(frame: &Frame) -> ([f64; 3], [f64; 3]) {
        let (sum, square, count) = frame
            .rgb
            .par_chunks_exact(3)
            .fold(
                || ([0.0; 3], [0.0; 3], 0u64),
                |(mut sum, mut square, count), pixel| {
                    for channel in 0..3 {
                        let value = pixel[channel] as f64;
                        sum[channel] += value;
                        square[channel] += value * value;
                    }
                    (sum, square, count + 1)
                },
            )
            .reduce(
                || ([0.0; 3], [0.0; 3], 0u64),
                |(mut a_sum, mut a_square, a_count), (b_sum, b_square, b_count)| {
                    for channel in 0..3 {
                        a_sum[channel] += b_sum[channel];
                        a_square[channel] += b_square[channel];
                    }
                    (a_sum, a_square, a_count + b_count)
                },
            );
        let count = count.max(1) as f64;
        let mut mean = [0.0; 3];
        let mut deviation = [1.0; 3];
        for channel in 0..3 {
            mean[channel] = sum[channel] / count;
            deviation[channel] = (square[channel] / count - mean[channel] * mean[channel])
                .max(1.0)
                .sqrt();
        }
        (mean, deviation)
    }

    let (model_mean, model_deviation) = stats(model_out);
    let (source_mean, source_deviation) = stats(source);
    let mut rgb = model_out.rgb.clone();
    rgb.par_chunks_mut(3).for_each(|pixel| {
        for channel in 0..3 {
            let gain = (source_deviation[channel] / model_deviation[channel]).clamp(0.8, 1.25);
            let corrected =
                (pixel[channel] as f64 - model_mean[channel]) * gain + source_mean[channel];
            pixel[channel] = corrected.clamp(0.0, 255.0) as u8;
        }
    });

    Frame {
        width: model_out.width,
        height: model_out.height,
        rgb,
    }
}

/// Caps how far the model may push any pixel away from the source.
///
/// This is the useful anti-hallucination control for GAN upscalers. A linear
/// blend attenuates *everything* equally, so pulling it down to kill an
/// artifact also throws away the sharpening you wanted. Structural mistakes —
/// freckles redrawn as black strokes — are large excursions from the source,
/// while genuine sharpening is a small one, so clamping the deviation removes
/// the former and leaves the latter alone.
///
/// `limit` is in 0..255 levels; 255 disables it.
pub fn clamp_deviation(model_out: &Frame, source: &Frame, limit: f32) -> Frame {
    if limit >= 254.0 {
        return Frame {
            width: model_out.width,
            height: model_out.height,
            rgb: model_out.rgb.clone(),
        };
    }

    let base = resample(source, model_out.width, model_out.height);
    let mut rgb = model_out.rgb.clone();
    rgb.par_iter_mut()
        .zip(base.rgb.par_iter())
        .for_each(|(pixel, &base_pixel)| {
            let b = base_pixel as f32;
            let m = *pixel as f32;
            *pixel = m.clamp(b - limit, b + limit).clamp(0.0, 255.0) as u8;
        });

    Frame {
        width: model_out.width,
        height: model_out.height,
        rgb,
    }
}

/// Area-average resample. Used to bring a model's native scale down to the
/// requested one — averaging is deliberate, since it is what suppresses the
/// invented high-frequency detail that a nearest or bilinear shrink keeps.
pub fn resample(src: &Frame, dw: u32, dh: u32) -> Frame {
    if src.width == dw && src.height == dh {
        return Frame {
            width: dw,
            height: dh,
            rgb: src.rgb.clone(),
        };
    }

    let (sw, sh) = (src.width as usize, src.height as usize);
    let mut rgb = vec![0u8; (dw as usize) * (dh as usize) * 3];
    let xr = sw as f32 / dw as f32;
    let yr = sh as f32 / dh as f32;

    rgb.par_chunks_mut(dw as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y0 = ((y as f32) * yr).floor() as usize;
            let y1 = (((y as f32 + 1.0) * yr).ceil() as usize)
                .min(sh)
                .max(y0 + 1);
            for (x, pixel) in row.chunks_mut(3).enumerate() {
                let x0 = ((x as f32) * xr).floor() as usize;
                let x1 = (((x as f32 + 1.0) * xr).ceil() as usize)
                    .min(sw)
                    .max(x0 + 1);

                let n = ((x1 - x0) * (y1 - y0)) as f32;
                let mut acc = [0f32; 3];
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        let si = (sy * sw + sx) * 3;
                        acc[0] += src.rgb[si] as f32;
                        acc[1] += src.rgb[si + 1] as f32;
                        acc[2] += src.rgb[si + 2] as f32;
                    }
                }
                for c in 0..3 {
                    pixel[c] = (acc[c] / n).clamp(0.0, 255.0) as u8;
                }
            }
        });

    Frame {
        width: dw,
        height: dh,
        rgb,
    }
}

/// Feathers a planar CHW tile into the packed RGB accumulator.
#[allow(clippy::too_many_arguments)]
fn blend_into(
    accum: &mut [f32],
    weight: &mut [f32],
    dw: u32,
    dh: u32,
    tile: &[f32],
    ox: u32,
    oy: u32,
    tw: u32,
    th: u32,
    feather: u32,
) {
    let plane = (tw as usize) * (th as usize);
    let feather = feather.max(1) as f32;

    for ty in 0..th {
        let gy = oy + ty;
        if gy >= dh {
            break;
        }
        // Ramp weight down towards every tile edge so neighbours cross-fade.
        let wy = ((ty as f32 + 0.5) / feather)
            .min((th - ty) as f32 / feather)
            .clamp(0.02, 1.0);

        for tx in 0..tw {
            let gx = ox + tx;
            if gx >= dw {
                break;
            }
            let wx = ((tx as f32 + 0.5) / feather)
                .min((tw - tx) as f32 / feather)
                .clamp(0.02, 1.0);
            let w = wx * wy;

            let si = (ty as usize) * (tw as usize) + tx as usize;
            let di = (gy as usize) * (dw as usize) + gx as usize;

            accum[di * 3] += tile[si] * w;
            accum[di * 3 + 1] += tile[plane + si] * w;
            accum[di * 3 + 2] += tile[plane * 2 + si] * w;
            weight[di] += w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, scale: u32) -> ModelEntry {
        ModelEntry {
            name: name.into(),
            file: format!("{name}.onnx"),
            fp16_file: None,
            scale,
            architecture: "rrdb".into(),
            temporal_frames: 1,
        }
    }

    #[test]
    fn scale_advice_warns_only_when_the_model_overshoots() {
        let installed = vec![entry("Big", 4), entry("Small", 2)];

        // Matching scale: nothing to say.
        assert!(scale_advice(&entry("Small", 2), 2, &installed).is_none());
        // Model below target still runs once; no warning.
        assert!(scale_advice(&entry("Small", 2), 4, &installed).is_none());
        // Unknown scale is probed later, so stay quiet.
        assert!(scale_advice(&entry("Unknown", 0), 2, &installed).is_none());

        let advice = scale_advice(&entry("Big", 4), 2, &installed).expect("warns");
        assert!(advice.contains("~4x the work"), "{advice}");
        assert!(
            advice.contains("Small"),
            "should name the native option: {advice}"
        );
    }

    #[test]
    fn scale_advice_omits_alternative_when_none_installed() {
        let installed = vec![entry("Big", 4)];
        let advice = scale_advice(&entry("Big", 4), 2, &installed).expect("warns");
        assert!(advice.contains("~4x the work"), "{advice}");
        assert!(!advice.contains(" — "), "no alternative to name: {advice}");
    }

    #[test]
    fn color_match_corrects_channel_drift() {
        let source = Frame {
            width: 2,
            height: 1,
            rgb: vec![80, 100, 120, 120, 140, 160],
        };
        let shifted = Frame {
            width: 2,
            height: 1,
            rgb: vec![100, 120, 140, 140, 160, 180],
        };
        let matched = match_color_to_source(&shifted, &source);
        assert_eq!(matched.rgb, source.rgb);
    }
}
