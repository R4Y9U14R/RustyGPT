use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(&manifest_dir).join("libs");
    let cuda_file = "src/common/cuda/dense.cu";

    fs::create_dir_all(&out_dir).expect("Failed to create directory");

    let status = Command::new("nvcc")
        .args(&["-shared", "-o", &format!("{}/libdense.so", out_dir.display()), "-Xcompiler", "-fPIC", cuda_file])
        .status()
        .expect("Failed to compile CUDA file.");

    if !status.success() {
        panic!("nvcc failed to compile the CUDA file");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=dense");
    println!("cargo:rerun-if-changed={}", cuda_file);
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}", out_dir.display());
}
