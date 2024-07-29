use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(&manifest_dir).join("libs");
    let cuda_file = "src/common/cuda/dense.cu";

    fs::create_dir_all(&out_dir).expect("Failed to create directory");

    let status: Option<ExitStatus>;

    #[cfg(any(target_os = "linux", target_os = "macos"))]  // unix-like
    {
        status = Some(Command::new("nvcc")
            .args(&[
                "-shared",
                "-o", &format!("{}/libdense.so", out_dir.display()),
                "-Xcompiler",
                "-fPIC",
                cuda_file
            ])
            .status()
            .expect("Failed to compile CUDA file."));
    }

    #[cfg(target_os = "windows")]  // windows
    {
        status = Some(Command::new("nvcc")
            .args(&[
                "-DBUILD_DLL",
                &format!("-I{}", "src/common/cuda").to_string(),
                "-shared",
                "-o", &format!("{}/dense.dll", out_dir.display()),
                cuda_file,
            ])
            .status()
            .expect("Failed to compile CUDA file."));
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]  // idk if it'll work on other os
    {
        eprintln!("Sorry, your operating system is currently not supported.");
    }

    if let Some(status) = status {
        if !status.success() {
            panic!("nvcc failed to compile the CUDA file.");
        } else {
            println!("Compliation completed.");
        }
    } else {
        eprintln!("No compilation was attempted.");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=dense");
    println!("cargo:rerun-if-changed={}", cuda_file);
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-env=PATH={};{}", env::var("PATH").unwrap(), out_dir.display())
    } else {
        println!("cargo:rustc-env=LD_LIBRARY_PATH={}", out_dir.display());
    }
}
