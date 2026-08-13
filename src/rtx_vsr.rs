//! Native NVIDIA RTX Video Super Resolution for still images.
//!
//! This calls NVIDIA's Video Effects SDK directly. It does not start ComfyUI,
//! load a scripting runtime, or compile a model. The proprietary SDK DLLs are
//! an optional, colocated runtime under `rtx-vsr/`.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use libloading::Library;

use crate::image_mode;
use crate::media::Frame;
use crate::upscale;

const VERIFIED_MAX_EDGE: u32 = 15_360;
const MAX_OUTPUT_EDGE: u32 = 16_384;

const DLLS: &[&str] = &[
    "nppc64_12.dll",
    "nppial64_12.dll",
    "nppicc64_12.dll",
    "nppidei64_12.dll",
    "nppif64_12.dll",
    "nppig64_12.dll",
    "nppim64_12.dll",
    "nppist64_12.dll",
    "nppitc64_12.dll",
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll",
    "NVCVImage.dll",
    "nvngxruntime.dll",
    "NVVideoEffects.dll",
    "nvVFXVideoSuperRes.dll",
    "nvngx_vsr.dll",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Low = 1,
    Medium = 2,
    High = 3,
    Ultra = 4,
}

impl Quality {
    pub fn from_index(index: i32) -> Self {
        match index {
            0 => Self::Low,
            1 => Self::Medium,
            2 => Self::High,
            _ => Self::Ultra,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Ultra => "Ultra",
        }
    }
}

#[derive(Clone)]
pub struct ExportSpec {
    pub image: image_mode::ImageInfo,
    pub target_scale: u32,
    pub quality: Quality,
    pub out_path: PathBuf,
}

impl ExportSpec {
    pub fn output_size(&self) -> (u32, u32) {
        crate::resolution::dimensions(
            self.image.width,
            self.image.height,
            self.target_scale,
            false,
        )
    }
}

pub fn runtime_dir() -> Option<PathBuf> {
    let beside_exe = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("rtx-vsr")));
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("rtx-vsr");
    let override_dir = std::env::var_os("KRAKEN_RTX_VSR_DIR").map(PathBuf::from);
    [beside_exe, Some(workspace), override_dir]
        .into_iter()
        .flatten()
        .find(|dir| DLLS.iter().all(|name| dir.join(name).is_file()))
}

pub fn available() -> bool {
    runtime_dir().is_some()
}

pub fn run(
    spec: &ExportSpec,
    cancel: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    on_progress(0, 3);
    let source = image_mode::decode(&spec.image)?;
    on_progress(1, 3);
    let working_scale =
        crate::resolution::working_scale(source.width, source.height, spec.target_scale, false)?;
    let mut output = upscale_frame(&source, working_scale, spec.quality)?;
    let (width, height) = spec.output_size();
    if (output.width, output.height) != (width, height) {
        output = upscale::resample(&output, width, height);
    }
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }
    on_progress(2, 3);
    image_mode::write_output(&spec.image, &output, &spec.out_path)?;
    on_progress(3, 3);
    Ok(())
}

pub fn upscale_frame(source: &Frame, scale: u32, quality: Quality) -> Result<Frame, String> {
    let target_width = source
        .width
        .checked_mul(scale)
        .ok_or_else(|| "RTX VSR output width overflow".to_string())?;
    let target_height = source
        .height
        .checked_mul(scale)
        .ok_or_else(|| "RTX VSR output height overflow".to_string())?;
    if scale == 0 || target_width == 0 || target_height == 0 {
        return Err("RTX VSR output dimensions must be positive".into());
    }
    if target_width.max(target_height) > MAX_OUTPUT_EDGE {
        return Err(format!(
            "RTX VSR supports a maximum output edge of {MAX_OUTPUT_EDGE}px; requested {target_width}×{target_height}"
        ));
    }

    let (vsr_width, vsr_height) = if target_width.max(target_height) <= VERIFIED_MAX_EDGE {
        (target_width, target_height)
    } else {
        let ratio = VERIFIED_MAX_EDGE as f64 / target_width.max(target_height) as f64;
        (
            align_down(target_width as f64 * ratio),
            align_down(target_height as f64 * ratio),
        )
    };
    if vsr_width < source.width || vsr_height < source.height {
        return Err("RTX VSR protected intermediate would downscale the source".into());
    }

    let runtime = runtime_dir().ok_or_else(|| {
        "NVIDIA RTX VSR runtime missing — expected the rtx-vsr folder beside Kraken Upscale.exe"
            .to_string()
    })?;
    let enhanced = unsafe { Api::load(&runtime)?.process(source, vsr_width, vsr_height, quality) }?;
    assert_channel_integrity(source, &enhanced)?;
    if (vsr_width, vsr_height) == (target_width, target_height) {
        Ok(enhanced)
    } else {
        Ok(upscale::resample(&enhanced, target_width, target_height))
    }
}

fn align_down(value: f64) -> u32 {
    ((value.floor() as u32) / 8 * 8).max(8)
}

fn assert_channel_integrity(source: &Frame, output: &Frame) -> Result<(), String> {
    let means = |frame: &Frame| {
        let step = (frame.width.max(frame.height) / 512).max(1) as usize;
        let mut sum = [0u64; 3];
        let mut count = 0u64;
        for y in (0..frame.height as usize).step_by(step) {
            for x in (0..frame.width as usize).step_by(step) {
                let pixel = (y * frame.width as usize + x) * 3;
                for (channel, total) in sum.iter_mut().enumerate() {
                    *total += frame.rgb[pixel + channel] as u64;
                }
                count += 1;
            }
        }
        [
            sum[0] as f64 / count.max(1) as f64 / 255.0,
            sum[1] as f64 / count.max(1) as f64 / 255.0,
            sum[2] as f64 / count.max(1) as f64 / 255.0,
        ]
    };
    let input = means(source);
    let result = means(output);
    for channel in 0..3 {
        let others = result[(channel + 1) % 3].max(result[(channel + 2) % 3]);
        if input[channel] > 0.02 && result[channel] < 0.003 && others > 0.02 {
            let name = ["red", "green", "blue"][channel];
            return Err(format!(
                "NVIDIA RTX VSR returned a corrupted image: the {name} channel collapsed"
            ));
        }
    }
    Ok(())
}

#[repr(C)]
#[derive(Default)]
struct NvCvImage {
    width: u32,
    height: u32,
    pitch: i32,
    pixel_format: c_int,
    component_type: c_int,
    pixel_bytes: u8,
    component_bytes: u8,
    num_components: u8,
    planar: u8,
    gpu_mem: u8,
    colorspace: u8,
    reserved: [u8; 2],
    pixels: *mut c_void,
    delete_ptr: *mut c_void,
    delete_proc: Option<unsafe extern "C" fn(*mut c_void)>,
    buffer_bytes: u64,
}

type NvStatus = c_int;
type NvEffect = *mut c_void;
type NvImageInit = unsafe extern "C" fn(
    *mut NvCvImage,
    u32,
    u32,
    c_int,
    *mut c_void,
    c_int,
    c_int,
    u32,
    u32,
) -> NvStatus;
type NvImageAlloc =
    unsafe extern "C" fn(*mut NvCvImage, u32, u32, c_int, c_int, u32, u32, u32) -> NvStatus;
type NvImageDealloc = unsafe extern "C" fn(*mut NvCvImage);
type NvImageTransfer = unsafe extern "C" fn(
    *const NvCvImage,
    *mut NvCvImage,
    f32,
    *mut c_void,
    *mut NvCvImage,
) -> NvStatus;
type NvCreateEffect = unsafe extern "C" fn(*const c_char, *mut NvEffect) -> NvStatus;
type NvDestroyEffect = unsafe extern "C" fn(NvEffect);
type NvSetU32 = unsafe extern "C" fn(NvEffect, *const c_char, u32) -> NvStatus;
type NvSetImage = unsafe extern "C" fn(NvEffect, *const c_char, *mut NvCvImage) -> NvStatus;
type NvSetStream = unsafe extern "C" fn(NvEffect, *const c_char, *mut c_void) -> NvStatus;
type NvLoad = unsafe extern "C" fn(NvEffect) -> NvStatus;
type NvRun = unsafe extern "C" fn(NvEffect, c_int) -> NvStatus;
type NvErrorString = unsafe extern "C" fn(NvStatus) -> *const c_char;

struct Api {
    _libraries: Vec<Library>,
    image_init: NvImageInit,
    image_alloc: NvImageAlloc,
    image_dealloc: NvImageDealloc,
    image_transfer: NvImageTransfer,
    create_effect: NvCreateEffect,
    destroy_effect: NvDestroyEffect,
    set_u32: NvSetU32,
    set_image: NvSetImage,
    set_stream: NvSetStream,
    load_effect: NvLoad,
    run_effect: NvRun,
    error_string: NvErrorString,
}

impl Api {
    unsafe fn load(runtime: &Path) -> Result<Self, String> {
        let mut libraries = Vec::new();
        for name in DLLS {
            libraries.push(
                Library::new(runtime.join(name))
                    .map_err(|error| format!("loading NVIDIA RTX VSR {name}: {error}"))?,
            );
        }
        let image = libraries
            .iter()
            .find(|library| library.get::<NvImageAlloc>(b"NvCVImage_Alloc\0").is_ok())
            .ok_or_else(|| "NVCVImage.dll exports were not found".to_string())?;
        let effects = libraries
            .iter()
            .find(|library| {
                library
                    .get::<NvCreateEffect>(b"NvVFX_CreateEffect\0")
                    .is_ok()
            })
            .ok_or_else(|| "NVVideoEffects.dll exports were not found".to_string())?;

        Ok(Self {
            image_init: *image
                .get(b"NvCVImage_Init\0")
                .map_err(|error| error.to_string())?,
            image_alloc: *image
                .get(b"NvCVImage_Alloc\0")
                .map_err(|error| error.to_string())?,
            image_dealloc: *image
                .get(b"NvCVImage_Dealloc\0")
                .map_err(|error| error.to_string())?,
            image_transfer: *image
                .get(b"NvCVImage_Transfer\0")
                .map_err(|error| error.to_string())?,
            create_effect: *effects
                .get(b"NvVFX_CreateEffect\0")
                .map_err(|error| error.to_string())?,
            destroy_effect: *effects
                .get(b"NvVFX_DestroyEffect\0")
                .map_err(|error| error.to_string())?,
            set_u32: *effects
                .get(b"NvVFX_SetU32\0")
                .map_err(|error| error.to_string())?,
            set_image: *effects
                .get(b"NvVFX_SetImage\0")
                .map_err(|error| error.to_string())?,
            set_stream: *effects
                .get(b"NvVFX_SetCudaStream\0")
                .map_err(|error| error.to_string())?,
            load_effect: *effects
                .get(b"NvVFX_Load\0")
                .map_err(|error| error.to_string())?,
            run_effect: *effects
                .get(b"NvVFX_Run\0")
                .map_err(|error| error.to_string())?,
            error_string: *effects
                .get(b"NvCV_GetErrorStringFromCode\0")
                .or_else(|_| image.get(b"NvCV_GetErrorStringFromCode\0"))
                .map_err(|error| error.to_string())?,
            _libraries: libraries,
        })
    }

    unsafe fn process(
        &self,
        source: &Frame,
        output_width: u32,
        output_height: u32,
        quality: Quality,
    ) -> Result<Frame, String> {
        let mut input_rgba = vec![255u8; source.width as usize * source.height as usize * 4];
        for (source_pixel, target_pixel) in source
            .rgb
            .chunks_exact(3)
            .zip(input_rgba.chunks_exact_mut(4))
        {
            target_pixel[..3].copy_from_slice(source_pixel);
        }
        let mut output_rgba = vec![0u8; output_width as usize * output_height as usize * 4];
        let mut input_cpu = NvCvImage::default();
        let mut output_cpu = NvCvImage::default();
        let mut input_gpu = NvCvImage::default();
        let mut output_gpu = NvCvImage::default();
        let mut effect: NvEffect = std::ptr::null_mut();

        let result: Result<(), String> = (|| {
            self.check(
                (self.image_init)(
                    &mut input_cpu,
                    source.width,
                    source.height,
                    (source.width * 4) as c_int,
                    input_rgba.as_mut_ptr().cast(),
                    6,
                    1,
                    0,
                    0,
                ),
                "initializing RTX VSR input",
            )?;
            self.check(
                (self.image_init)(
                    &mut output_cpu,
                    output_width,
                    output_height,
                    (output_width * 4) as c_int,
                    output_rgba.as_mut_ptr().cast(),
                    6,
                    1,
                    0,
                    0,
                ),
                "initializing RTX VSR output",
            )?;
            self.check(
                (self.image_alloc)(&mut input_gpu, source.width, source.height, 6, 1, 0, 1, 1),
                "allocating RTX VSR input on the GPU",
            )?;
            self.check(
                (self.image_alloc)(&mut output_gpu, output_width, output_height, 6, 1, 0, 1, 1),
                "allocating RTX VSR output on the GPU",
            )?;
            self.check(
                (self.image_transfer)(
                    &input_cpu,
                    &mut input_gpu,
                    1.0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
                "uploading RTX VSR input",
            )?;
            self.check(
                (self.create_effect)(c"VideoSuperRes".as_ptr(), &mut effect),
                "creating NVIDIA VideoSuperRes",
            )?;
            self.check(
                (self.set_u32)(effect, c"QualityLevel".as_ptr(), quality as u32),
                "setting RTX VSR quality",
            )?;
            self.check(
                (self.set_image)(effect, c"SrcImage0".as_ptr(), &mut input_gpu),
                "setting RTX VSR input",
            )?;
            self.check(
                (self.set_image)(effect, c"DstImage0".as_ptr(), &mut output_gpu),
                "setting RTX VSR output",
            )?;
            self.check(
                (self.set_stream)(effect, c"CudaStream".as_ptr(), std::ptr::null_mut()),
                "setting RTX VSR CUDA stream",
            )?;
            self.check((self.load_effect)(effect), "loading NVIDIA VideoSuperRes")?;
            self.check((self.run_effect)(effect, 0), "running NVIDIA VideoSuperRes")?;
            self.check(
                (self.image_transfer)(
                    &output_gpu,
                    &mut output_cpu,
                    1.0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
                "downloading RTX VSR output",
            )?;
            Ok(())
        })();

        if !effect.is_null() {
            (self.destroy_effect)(effect);
        }
        if !input_gpu.delete_ptr.is_null() {
            (self.image_dealloc)(&mut input_gpu);
        }
        if !output_gpu.delete_ptr.is_null() {
            (self.image_dealloc)(&mut output_gpu);
        }
        result?;

        let mut rgb = Vec::with_capacity(output_width as usize * output_height as usize * 3);
        for pixel in output_rgba.chunks_exact(4) {
            rgb.extend_from_slice(&pixel[..3]);
        }
        Ok(Frame {
            width: output_width,
            height: output_height,
            rgb,
        })
    }

    fn check(&self, status: NvStatus, operation: &str) -> Result<(), String> {
        if status == 0 {
            return Ok(());
        }
        let detail = unsafe {
            let pointer = (self.error_string)(status);
            if pointer.is_null() {
                "unknown NVIDIA error".into()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Err(format!("{operation}: {detail} (code {status})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_indices_default_to_ultra() {
        assert_eq!(Quality::from_index(0), Quality::Low);
        assert_eq!(Quality::from_index(2), Quality::High);
        assert_eq!(Quality::from_index(99), Quality::Ultra);
    }

    #[test]
    fn protected_intermediate_alignment_does_not_exceed_limit() {
        let ratio = VERIFIED_MAX_EDGE as f64 / 16_384.0;
        assert_eq!(align_down(16_384.0 * ratio), VERIFIED_MAX_EDGE);
        assert_eq!(align_down(9_216.0 * ratio), 8_640);
    }

    #[test]
    #[ignore = "requires the optional NVIDIA RTX VSR runtime and a supported GPU"]
    fn hardware_smoke_processes_rgba_on_the_gpu() {
        let mut rgb = vec![0u8; 640 * 360 * 3];
        for y in 0..360usize {
            for x in 0..640usize {
                let pixel = (y * 640 + x) * 3;
                rgb[pixel] = (x % 256) as u8;
                rgb[pixel + 1] = (y % 256) as u8;
                rgb[pixel + 2] = ((x + y) % 256) as u8;
            }
        }
        let output = upscale_frame(
            &Frame {
                width: 640,
                height: 360,
                rgb,
            },
            2,
            Quality::Ultra,
        )
        .expect("RTX VSR hardware smoke");
        assert_eq!((output.width, output.height), (1280, 720));
        assert!(output.rgb.iter().any(|value| *value != 0));
    }
}
