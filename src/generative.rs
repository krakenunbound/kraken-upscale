//! Native Ultimate-SD-Upscale-style still-image refinement.
//!
//! Kraken Art's USDU pipeline is the behavioral reference, but this module
//! uses the bundled stable-diffusion.cpp CUDA server instead of a scripting
//! sidecar. Model packages are ordinary downloadable weights described by a
//! small JSON manifest under `models/generative/`.

use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use serde::Deserialize;

use crate::image_mode;
use crate::media::Frame;
use crate::upscale::{self, Engine};

const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(12 * 60);
const HTTP_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const TILE_CONTEXT: u32 = 64;

#[derive(Clone, Debug)]
pub struct ModelPackage {
    pub name: String,
    pub architecture: String,
    pub diffusion_model: PathBuf,
    pub vae: PathBuf,
    pub clip_l: PathBuf,
    pub t5xxl: PathBuf,
    pub guidance: f32,
}

#[derive(Deserialize)]
struct ModelManifest {
    name: String,
    #[serde(default = "default_architecture")]
    architecture: String,
    diffusion_model: String,
    vae: String,
    clip_l: String,
    t5xxl: String,
    #[serde(default = "default_guidance")]
    guidance: f32,
}

fn default_architecture() -> String {
    "flux1".into()
}

fn default_guidance() -> f32 {
    3.5
}

#[derive(Clone)]
pub struct ExportSpec {
    pub conventional: image_mode::ExportSpec,
    pub package: ModelPackage,
    pub prompt: String,
    pub steps: u32,
    pub denoise: f32,
    pub seed: i64,
    pub refine_tile: u32,
    pub quality_finish: bool,
}

impl ExportSpec {
    pub fn output_size(&self) -> (u32, u32) {
        self.conventional.output_size()
    }

    pub fn out_path(&self) -> &Path {
        &self.conventional.out_path
    }
}

pub fn models_dir() -> PathBuf {
    upscale::models_dir().join("generative")
}

pub fn runtime_dir() -> Option<PathBuf> {
    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("native-diffusion")));
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("native-diffusion");
    let staging = Path::new(env!("CARGO_MANIFEST_DIR")).join(".native-staging/sd-cuda");
    [beside_exe, Some(workspace), Some(staging)]
        .into_iter()
        .flatten()
        .find(|dir| dir.join("sd-server.exe").is_file())
}

pub fn runtime_available() -> bool {
    runtime_dir().is_some()
}

pub fn registry() -> Vec<ModelPackage> {
    let root = models_dir();
    let Ok(entries) = fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut manifests = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            manifests.push(path);
        }
    }

    let mut packages = Vec::new();
    for path in manifests {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(manifest) = serde_json::from_str::<ModelManifest>(&text) else {
            continue;
        };
        let base = path.parent().unwrap_or(&root);
        let resolve = |value: &str| {
            let candidate = PathBuf::from(value);
            if candidate.is_absolute() {
                candidate
            } else {
                base.join(candidate)
            }
        };
        let package = ModelPackage {
            name: manifest.name,
            architecture: manifest.architecture,
            diffusion_model: resolve(&manifest.diffusion_model),
            vae: resolve(&manifest.vae),
            clip_l: resolve(&manifest.clip_l),
            t5xxl: resolve(&manifest.t5xxl),
            guidance: manifest.guidance,
        };
        if [
            &package.diffusion_model,
            &package.vae,
            &package.clip_l,
            &package.t5xxl,
        ]
        .iter()
        .all(|path| path.is_file())
        {
            packages.push(package);
        }
    }
    packages.sort_by_key(|package| package.name.to_lowercase());
    packages
}

/// First pass: run the selected faithful upscaler and return its exact target
/// canvas without writing it. The caller drops the ONNX session before loading
/// the much larger diffusion runtime.
pub fn pre_upscale(
    spec: &ExportSpec,
    engine: &mut Engine,
    overlap: u32,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<Frame, String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    let source = image_mode::decode(&spec.conventional.image)?;
    let mut progress = |done: u32, total: u32| on_progress(done as u64, total as u64);
    let output = engine
        .run_batch(
            std::slice::from_ref(&source),
            spec.conventional.tile,
            overlap,
            &mut progress,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| "image model returned no frame".to_string())?;
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    Ok(image_mode::finish(&spec.conventional, output, &source))
}

/// Second pass: refine overlapping image tiles through a single resident native
/// diffusion context, blend them, snap to the exact target, and write PNG.
pub fn refine(
    spec: &ExportSpec,
    canvas: Frame,
    cancel: Arc<AtomicBool>,
    on_progress: &mut dyn FnMut(f32, &str),
) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }

    let (target_w, target_h) = spec.output_size();
    let pred_w = align16(target_w);
    let pred_h = align16(target_h);
    let canvas = if canvas.width != pred_w || canvas.height != pred_h {
        upscale::resample(&canvas, pred_w, pred_h)
    } else {
        canvas
    };

    on_progress(0.0, "loading native generative model");
    let mut server = NativeServer::start(&spec.package, cancel.as_ref())?;

    let tile = spec.refine_tile.clamp(512, 1536).next_multiple_of(64);
    let cells = tile_grid(pred_w, pred_h, tile);
    let total = cells.len() as u32;
    let refine_started = Instant::now();
    let mut accum = vec![0f32; pred_w as usize * pred_h as usize * 3];
    let mut weights = vec![0f32; pred_w as usize * pred_h as usize];

    for (index, cell) in cells.iter().copied().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let region = padded_region(cell, pred_w, pred_h, TILE_CONTEXT);
        let crop = crop_frame(&canvas, region);
        let work_w = align16(crop.width);
        let work_h = align16(crop.height);
        let work = if work_w != crop.width || work_h != crop.height {
            upscale::resample(&crop, work_w, work_h)
        } else {
            crop
        };

        let message = format!("tile {}/{} · preparing", index + 1, total);
        on_progress(index as f32 / total.max(1) as f32, &message);
        let tile_seed = spatial_seed(spec.seed, cell.0, cell.1);
        let mut tile_progress = |step: u32, step_total: u32, tile_elapsed: Duration| {
            let within = if step_total == 0 {
                0.0
            } else {
                step as f32 / step_total as f32
            };
            let overall = (index as f32 + within) / total.max(1) as f32;
            let elapsed = refine_started.elapsed();
            let eta = if overall > 0.001 {
                let remaining = elapsed.mul_f64((1.0 - overall) as f64 / overall as f64);
                format!(" · ~{} left", compact_duration(remaining))
            } else {
                String::new()
            };
            let phase = if step_total == 0 {
                "preparing".to_string()
            } else {
                format!("step {step}/{step_total}")
            };
            let message = format!(
                "tile {}/{} · {phase} · {} elapsed{eta}",
                index + 1,
                total,
                compact_duration(tile_elapsed)
            );
            on_progress(overall, &message);
        };
        let mut refined = server.refine_tile(
            &work,
            &spec.prompt,
            spec.steps.clamp(1, 60),
            spec.denoise.clamp(0.01, 0.65),
            tile_seed,
            spec.package.guidance,
            cancel.clone(),
            &mut tile_progress,
        )?;
        if refined.width != region.2 - region.0 || refined.height != region.3 - region.1 {
            refined = upscale::resample(&refined, region.2 - region.0, region.3 - region.1);
        }
        blend_region(
            &mut accum,
            &mut weights,
            pred_w,
            pred_h,
            &refined,
            region,
            TILE_CONTEXT,
        );
        let complete = format!("tile {}/{} complete", index + 1, total);
        on_progress((index + 1) as f32 / total.max(1) as f32, &complete);
    }

    let mut refined = finish_blend(&canvas, accum, weights);
    if spec.conventional.match_color {
        refined = upscale::match_color_to_source(&refined, &canvas);
    }
    if spec.quality_finish {
        refined = restrained_unsharp(&refined);
    }
    if refined.width != target_w || refined.height != target_h {
        refined = upscale::resample(&refined, target_w, target_h);
    }
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    image_mode::write_output(
        &spec.conventional.image,
        &refined,
        &spec.conventional.out_path,
    )
}

fn align16(value: u32) -> u32 {
    value.max(16).next_multiple_of(16)
}

type Rect = (u32, u32, u32, u32);

fn tile_grid(width: u32, height: u32, tile: u32) -> Vec<Rect> {
    let mut cells = Vec::new();
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            cells.push((x, y, (x + tile).min(width), (y + tile).min(height)));
            x += tile;
        }
        y += tile;
    }
    cells
}

fn padded_region(cell: Rect, width: u32, height: u32, padding: u32) -> Rect {
    (
        cell.0.saturating_sub(padding),
        cell.1.saturating_sub(padding),
        (cell.2 + padding).min(width),
        (cell.3 + padding).min(height),
    )
}

fn crop_frame(frame: &Frame, rect: Rect) -> Frame {
    let width = rect.2 - rect.0;
    let height = rect.3 - rect.1;
    let mut rgb = vec![0u8; width as usize * height as usize * 3];
    for y in 0..height as usize {
        let src = ((rect.1 as usize + y) * frame.width as usize + rect.0 as usize) * 3;
        let dst = y * width as usize * 3;
        rgb[dst..dst + width as usize * 3]
            .copy_from_slice(&frame.rgb[src..src + width as usize * 3]);
    }
    Frame { width, height, rgb }
}

fn edge_weight(position: u32, length: u32, at_start: bool, at_end: bool, fade: u32) -> f32 {
    let mut weight = 1.0f32;
    if !at_start && position < fade {
        let phase = position as f32 / fade.max(1) as f32;
        weight *= 0.5 - 0.5 * (std::f32::consts::PI * phase).cos();
    }
    let from_end = length.saturating_sub(position + 1);
    if !at_end && from_end < fade {
        let phase = from_end as f32 / fade.max(1) as f32;
        weight *= 0.5 - 0.5 * (std::f32::consts::PI * phase).cos();
    }
    weight.max(1e-4)
}

fn blend_region(
    accum: &mut [f32],
    weights: &mut [f32],
    canvas_w: u32,
    canvas_h: u32,
    tile: &Frame,
    region: Rect,
    fade: u32,
) {
    let at_left = region.0 == 0;
    let at_top = region.1 == 0;
    let at_right = region.2 == canvas_w;
    let at_bottom = region.3 == canvas_h;
    for y in 0..tile.height {
        let wy = edge_weight(y, tile.height, at_top, at_bottom, fade);
        for x in 0..tile.width {
            let wx = edge_weight(x, tile.width, at_left, at_right, fade);
            let weight = wx * wy;
            let dst_pixel = (region.1 + y) as usize * canvas_w as usize + (region.0 + x) as usize;
            let src_pixel = (y as usize * tile.width as usize + x as usize) * 3;
            weights[dst_pixel] += weight;
            for channel in 0..3 {
                accum[dst_pixel * 3 + channel] += tile.rgb[src_pixel + channel] as f32 * weight;
            }
        }
    }
}

fn finish_blend(source: &Frame, accum: Vec<f32>, weights: Vec<f32>) -> Frame {
    let mut rgb = source.rgb.clone();
    for (index, weight) in weights.into_iter().enumerate() {
        if weight <= 1e-6 {
            continue;
        }
        for channel in 0..3 {
            rgb[index * 3 + channel] =
                (accum[index * 3 + channel] / weight).clamp(0.0, 255.0) as u8;
        }
    }
    Frame {
        width: source.width,
        height: source.height,
        rgb,
    }
}

fn restrained_unsharp(frame: &Frame) -> Frame {
    if frame.width < 3 || frame.height < 3 {
        return frame.clone();
    }
    let mut rgb = frame.rgb.clone();
    for y in 1..frame.height - 1 {
        for x in 1..frame.width - 1 {
            let center = (y as usize * frame.width as usize + x as usize) * 3;
            for channel in 0..3 {
                let mut sum = 0u32;
                for oy in y - 1..=y + 1 {
                    for ox in x - 1..=x + 1 {
                        let pixel = (oy as usize * frame.width as usize + ox as usize) * 3;
                        sum += frame.rgb[pixel + channel] as u32;
                    }
                }
                let blurred = sum as f32 / 9.0;
                let original = frame.rgb[center + channel] as f32;
                rgb[center + channel] =
                    (original + 0.35 * (original - blurred)).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Frame {
        width: frame.width,
        height: frame.height,
        rgb,
    }
}

fn spatial_seed(seed: i64, x: u32, y: u32) -> i64 {
    seed.wrapping_add((x as i64).wrapping_mul(73_856_093))
        .wrapping_add((y as i64).wrapping_mul(19_349_663))
}

fn compact_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn diffusion_progress(log: &str, requested_steps: u32) -> Option<(u32, u32)> {
    let generation = log.rsplit_once("generating image:")?.1;
    generation
        .split(['\r', '\n'])
        .filter_map(|line| {
            if !line.contains('|') {
                return None;
            }
            let before_rate = line.split_once(" - ")?.0;
            let ratio = before_rate.split_whitespace().last()?;
            let (done, total) = ratio.split_once('/')?;
            let done = done.parse::<u32>().ok()?;
            let total = total.parse::<u32>().ok()?;
            (total > 0 && total <= requested_steps && done <= total).then_some((done, total))
        })
        .next_back()
}

fn native_prompt(prompt: &str, guidance: f32) -> String {
    let prompt = if prompt.trim().is_empty() {
        "high detail, faithful textures, preserve composition"
    } else {
        prompt.trim()
    };
    // The SD-WebUI-compatible route only reads native-only settings from this
    // prompt envelope. FLUX uses distilled guidance rather than CFG.
    let native_args = serde_json::json!({
        "sample_params": { "guidance": { "distilled_guidance": guidance } }
    });
    format!("{prompt}<sd_cpp_extra_args>{native_args}</sd_cpp_extra_args>")
}

struct NativeServer {
    child: Child,
    port: u16,
    log_path: PathBuf,
}

impl NativeServer {
    fn start(package: &ModelPackage, cancel: &AtomicBool) -> Result<Self, String> {
        let model_bytes = fs::metadata(&package.diffusion_model)
            .map(|metadata| metadata.len())
            .unwrap_or_default();
        if crate::gpu::stats()
            .map(|stats| stats.vram_total_gb() <= 26.0)
            .unwrap_or(false)
            && model_bytes > 18 * 1024 * 1024 * 1024
        {
            return Err(format!(
                "{} is a {:.1} GB diffusion model and is too large for a 24 GB GPU; install the Q6 GGUF package instead",
                package.name,
                model_bytes as f64 / 1024.0_f64.powi(3)
            ));
        }
        let runtime = runtime_dir().ok_or_else(|| {
            "native generative runtime missing — expected native-diffusion/sd-server.exe beside Kraken"
                .to_string()
        })?;
        let port = TcpListener::bind(("127.0.0.1", 0))
            .and_then(|listener| listener.local_addr())
            .map_err(|error| format!("choosing local generative port: {error}"))?
            .port();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let log_path = std::env::temp_dir().join(format!("kraken-native-diffusion-{stamp}.log"));
        let stdout = File::create(&log_path)
            .map_err(|error| format!("creating native runtime log: {error}"))?;
        let stderr = stdout
            .try_clone()
            .map_err(|error| format!("opening native runtime diagnostics: {error}"))?;

        let mut command = Command::new(runtime.join("sd-server.exe"));
        command
            .current_dir(&runtime)
            .arg("--diffusion-model")
            .arg(&package.diffusion_model)
            .arg("--vae")
            .arg(&package.vae)
            .arg("--clip_l")
            .arg(&package.clip_l)
            .arg("--t5xxl")
            .arg(&package.t5xxl)
            .arg("--listen-ip")
            .arg("127.0.0.1")
            .arg("--listen-port")
            .arg(port.to_string())
            .arg("--auto-fit")
            .arg("--max-vram")
            .arg("-2")
            .arg("--stream-layers")
            .arg("--diffusion-fa")
            .arg("--vae-tiling")
            .arg("--rng")
            .arg("cpu")
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
        let current_path = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![runtime.clone()];
        paths.extend(std::env::split_paths(&current_path));
        if let Ok(path) = std::env::join_paths(paths) {
            command.env("PATH", path);
        }
        hide_console(&mut command);

        let child = command
            .spawn()
            .map_err(|error| format!("starting native generative runtime: {error}"))?;
        let mut server = Self {
            child,
            port,
            log_path,
        };
        let started = Instant::now();
        loop {
            if cancel.load(Ordering::Relaxed) {
                server.stop();
                return Err("cancelled".into());
            }
            if let Some(status) = server
                .child
                .try_wait()
                .map_err(|error| format!("checking native runtime: {error}"))?
            {
                return Err(format!(
                    "native generative runtime exited during model load ({status}): {}",
                    server.log_tail()
                ));
            }
            if http_request(
                port,
                "GET",
                "/sdapi/v1/sd-models",
                None,
                Duration::from_secs(2),
            )
            .is_ok()
            {
                return Ok(server);
            }
            if started.elapsed() > MODEL_LOAD_TIMEOUT {
                return Err(format!(
                    "native generative model load timed out: {}",
                    server.log_tail()
                ));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_tile(
        &mut self,
        tile: &Frame,
        prompt: &str,
        steps: u32,
        denoise: f32,
        seed: i64,
        guidance: f32,
        cancel: Arc<AtomicBool>,
        on_progress: &mut dyn FnMut(u32, u32, Duration),
    ) -> Result<Frame, String> {
        let png = encode_png(tile)?;
        let prompt = native_prompt(prompt, guidance);
        let request = serde_json::json!({
            "prompt": prompt,
            "negative_prompt": "",
            "init_images": [BASE64.encode(png)],
            "width": tile.width,
            "height": tile.height,
            "steps": steps,
            "cfg_scale": 1.0,
            "seed": seed,
            "batch_size": 1,
            "sampler_name": "Euler",
            "scheduler": "simple",
            "denoising_strength": denoise
        });
        let body = serde_json::to_vec(&request)
            .map_err(|error| format!("encoding generative request: {error}"))?;

        let pid = self.child.id();
        let port = self.port;
        let started = Instant::now();
        let response = std::thread::scope(|scope| {
            let request = scope.spawn(|| {
                http_request(port, "POST", "/sdapi/v1/img2img", Some(&body), HTTP_TIMEOUT)
            });
            let mut cancelled = false;
            while !request.is_finished() {
                if cancel.load(Ordering::Relaxed) && !cancelled {
                    terminate_process_tree(pid);
                    cancelled = true;
                }
                let progress = fs::read_to_string(&self.log_path)
                    .ok()
                    .and_then(|log| diffusion_progress(&log, steps))
                    .unwrap_or((0, 0));
                on_progress(progress.0, progress.1, started.elapsed());
                std::thread::sleep(Duration::from_millis(500));
            }
            request
                .join()
                .map_err(|_| "native generative request thread failed".to_string())?
        });
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let response = response.map_err(|error| {
            format!(
                "native generative tile failed: {error}: {}",
                self.log_tail()
            )
        })?;
        let json: serde_json::Value = serde_json::from_slice(&response)
            .map_err(|error| format!("reading generative response: {error}"))?;
        let encoded = json
            .get("images")
            .and_then(|images| images.as_array())
            .and_then(|images| images.first())
            .and_then(|image| image.as_str())
            .ok_or_else(|| format!("generative response contained no image: {json}"))?;
        let encoded = encoded
            .split_once(',')
            .map(|(_, data)| data)
            .unwrap_or(encoded);
        let bytes = BASE64
            .decode(encoded)
            .map_err(|error| format!("decoding generative image: {error}"))?;
        decode_png(&bytes)
    }

    fn log_tail(&self) -> String {
        let Ok(text) = fs::read_to_string(&self.log_path) else {
            return "native runtime produced no readable diagnostics".into();
        };
        let lines: Vec<_> = text.lines().rev().take(12).collect();
        lines.into_iter().rev().collect::<Vec<_>>().join(" | ")
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for NativeServer {
    fn drop(&mut self) {
        self.stop();
        let _ = fs::remove_file(&self.log_path);
    }
}

fn http_request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&[u8]>,
    timeout: Duration,
) -> Result<Vec<u8>, String> {
    let mut stream = TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}")
            .parse()
            .map_err(|error| format!("invalid local runtime address: {error}"))?,
        Duration::from_secs(1),
    )
    .map_err(|error| format!("connecting to native runtime: {error}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|error| format!("setting native runtime timeout: {error}"))?;
    let payload = body.unwrap_or_default();
    let content_type = if body.is_some() {
        "Content-Type: application/json\r\n"
    } else {
        ""
    };
    let header = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{content_type}Content-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|_| stream.write_all(payload))
        .map_err(|error| format!("sending native runtime request: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("reading native runtime response: {error}"))?;
    let split = response
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .ok_or_else(|| "native runtime returned an invalid HTTP response".to_string())?;
    let headers = String::from_utf8_lossy(&response[..split]);
    let status = headers.lines().next().unwrap_or_default();
    if !status.contains(" 200 ") {
        return Err(format!(
            "native runtime HTTP {status}: {}",
            String::from_utf8_lossy(&response[split + 4..])
        ));
    }
    Ok(response[split + 4..].to_vec())
}

fn encode_png(frame: &Frame) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, frame.width, frame.height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|error| format!("creating tile PNG: {error}"))?;
        writer
            .write_image_data(&frame.rgb)
            .map_err(|error| format!("encoding tile PNG: {error}"))?;
    }
    Ok(bytes)
}

fn decode_png(bytes: &[u8]) -> Result<Frame, String> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("opening generative PNG: {error}"))?;
    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|error| format!("decoding generative PNG: {error}"))?;
    let data = &buffer[..info.buffer_size()];
    let rgb = match info.color_type {
        png::ColorType::Rgb => data.to_vec(),
        png::ColorType::Rgba => data
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect(),
        other => return Err(format!("unsupported generative PNG color type: {other:?}")),
    };
    Ok(Frame {
        width: info.width,
        height: info.height,
        rgb,
    })
}

#[cfg(windows)]
fn hide_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(0x0800_0000);
}

#[cfg(not(windows))]
fn hide_console(_command: &mut Command) {}

#[cfg(windows)]
fn terminate_process_tree(pid: u32) {
    let mut command = Command::new("taskkill.exe");
    command.args(["/PID", &pid.to_string(), "/T", "/F"]);
    hide_console(&mut command);
    let _ = command.status();
}

#[cfg(not(windows))]
fn terminate_process_tree(_pid: u32) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_grid_covers_odd_canvas() {
        let cells = tile_grid(2017, 3585, 768);
        assert_eq!(cells.first().copied(), Some((0, 0, 768, 768)));
        assert_eq!(cells.last().copied(), Some((1536, 3072, 2017, 3585)));
    }

    #[test]
    fn png_round_trip_preserves_rgb() {
        let frame = Frame {
            width: 2,
            height: 2,
            rgb: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30],
        };
        let decoded = decode_png(&encode_png(&frame).expect("encode")).expect("decode");
        assert_eq!(decoded.width, frame.width);
        assert_eq!(decoded.height, frame.height);
        assert_eq!(decoded.rgb, frame.rgb);
    }

    #[test]
    fn spatial_seeds_are_stable_and_distinct() {
        assert_eq!(spatial_seed(42, 0, 0), 42);
        assert_ne!(spatial_seed(42, 768, 0), spatial_seed(42, 0, 768));
        assert_eq!(spatial_seed(42, 768, 0), spatial_seed(42, 768, 0));
    }

    #[test]
    fn parses_live_diffusion_step_progress() {
        let log = "loading 219/219\r\n[INFO] generating image: 1/1 - seed 0\r\n  |====| 1/2 - 160.77s/it\r  |#####| 780/780 - 2.8GB/s\r  |#####| 25/25 - 1.2it/s\r";
        assert_eq!(diffusion_progress(log, 15), Some((1, 2)));
        assert_eq!(diffusion_progress("still loading tensors", 15), None);
    }

    #[test]
    fn native_prompt_carries_flux_distilled_guidance() {
        let prompt = native_prompt(" preserve the face ", 3.5);
        assert!(prompt.starts_with("preserve the face<sd_cpp_extra_args>"));
        assert!(prompt.contains(r#""distilled_guidance":3.5"#));
        assert!(prompt.ends_with("</sd_cpp_extra_args>"));
    }

    #[test]
    #[ignore = "requires the complete FLUX Q6 package, bundled native runtime, and CUDA GPU"]
    fn runs_real_flux_tile_on_cuda() {
        let package = registry().into_iter().next().expect("FLUX package");
        let cancel = Arc::new(AtomicBool::new(false));
        let mut server = NativeServer::start(&package, cancel.as_ref()).expect("server start");
        let frame = Frame {
            width: 832,
            height: 832,
            rgb: vec![96; 832 * 832 * 3],
        };
        let started = Instant::now();
        let result = server
            .refine_tile(
                &frame,
                "faithful subtle photographic detail",
                15,
                0.10,
                42,
                package.guidance,
                cancel,
                &mut |done, total, elapsed| {
                    if total > 0 {
                        eprintln!(
                            "FLUX step {done}/{total} after {}",
                            compact_duration(elapsed)
                        );
                    }
                },
            )
            .expect("FLUX tile");
        eprintln!("FLUX 832px tile completed in {:?}", started.elapsed());
        assert_eq!((result.width, result.height), (832, 832));
        let log = fs::read_to_string(&server.log_path).expect("native runtime log");
        assert!(
            log.contains("CUDA"),
            "runtime log did not mention CUDA: {log}"
        );
    }
}
