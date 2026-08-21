//! Makes the CUDA runtime discoverable to ONNX Runtime.
//!
//! ORT's CUDA provider loads `cublasLt64_13.dll`, cuDNN 9 and friends by name.
//! Rather than demanding a system-wide CUDA toolkit install, candidate
//! directories are prepended to the process PATH before any session is built —
//! Windows' loader searches PATH, so this is enough to resolve them.

use std::path::PathBuf;

/// Directories searched, in order of preference.
fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    // 1. Vendored beside the executable — the shipping layout.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("cuda"));
        }
    }
    out.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"));
    out.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/cuda"));

    // 2. An explicit override, for unusual installs.
    if let Ok(p) = std::env::var("KRAKEN_CUDA_DIR") {
        out.push(PathBuf::from(p));
    }

    // 3. A system CUDA toolkit.
    if let Ok(p) = std::env::var("CUDA_PATH") {
        out.push(PathBuf::from(p).join("bin"));
    }

    out
}

/// Returns the directory that supplied the CUDA runtime, if one was found.
pub fn prepare() -> Option<PathBuf> {
    // cublasLt is the largest and most version-specific of the set; if it is
    // present the rest of the CUDA 13 runtime almost certainly is too.
    let found = candidates()
        .into_iter()
        .find(|d| d.join("cublasLt64_13.dll").is_file())?;

    let path = std::env::var("PATH").unwrap_or_default();
    std::env::set_var("PATH", format!("{};{path}", found.display()));
    Some(found)
}
