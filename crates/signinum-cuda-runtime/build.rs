use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/j2k_encode_kernels.cu");
    println!("cargo:rerun-if-env-changed=NVCC");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let ptx = out_dir.join("j2k_encode_kernels.ptx");
    let source = PathBuf::from("src/j2k_encode_kernels.cu");
    let nvcc = env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());

    let compiled = Command::new(&nvcc)
        .args(["--ptx", "-O3", "--std=c++14"])
        .arg(&source)
        .arg("-o")
        .arg(&ptx)
        .status()
        .is_ok_and(|status| status.success());

    if compiled {
        let mut bytes = fs::read(&ptx).expect("read generated CUDA PTX");
        if bytes.last().copied() != Some(0) {
            bytes.push(0);
            fs::write(&ptx, bytes).expect("NUL-terminate generated CUDA PTX");
        }
    } else {
        fs::write(&ptx, b".version 7.0\n.target sm_52\n.address_size 64\n\0")
            .expect("write fallback CUDA PTX");
    }
}
