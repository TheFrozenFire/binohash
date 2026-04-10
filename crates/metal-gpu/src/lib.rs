#[cfg(target_os = "macos")]
mod gpu;

#[cfg(target_os = "macos")]
pub use gpu::*;

#[cfg(not(target_os = "macos"))]
compile_error!("metal-gpu crate requires macOS with Metal support");
