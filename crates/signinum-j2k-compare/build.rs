use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_grok)");
    println!("cargo:rerun-if-env-changed=SIGNINUM_GROK_ROOT");
    println!("cargo:rerun-if-env-changed=SIGNINUM_GROK_SOURCE");
    println!("cargo:rerun-if-changed=src/grok_shim.c");

    if let Some(config) = grok_config() {
        let staged_lib_dir = stage_grok_runtime(&config.lib_dir)
            .unwrap_or_else(|err| panic!("failed to stage Grok runtime libraries: {err}"));
        println!("cargo:rustc-cfg=have_grok");
        println!(
            "cargo:rustc-link-search=native={}",
            staged_lib_dir.display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            config.lib_dir.display()
        );
        println!("cargo:rustc-link-lib=dylib=grokj2k");
        #[cfg(target_os = "macos")]
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            staged_lib_dir.display()
        );

        cc::Build::new()
            .file("src/grok_shim.c")
            .include(config.source_include)
            .include(config.build_include)
            .warnings(false)
            .compile("signinum_j2k_grok_shim");
    }
}

fn stage_grok_runtime(lib_dir: &Path) -> Result<PathBuf, String> {
    let out_dir =
        PathBuf::from(std::env::var_os("OUT_DIR").ok_or_else(|| "OUT_DIR missing".to_string())?);
    #[cfg(target_os = "macos")]
    {
        stage_family(
            lib_dir,
            &out_dir,
            "libgrokj2k.20.3.0.dylib",
            "libgrokj2k.1.dylib",
            "libgrokj2k.dylib",
        )?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = lib_dir;
    }
    Ok(out_dir)
}

#[cfg(target_os = "macos")]
fn stage_family(
    lib_dir: &Path,
    out_dir: &Path,
    real_name: &str,
    compat_name: &str,
    link_name: &str,
) -> Result<(), String> {
    let real_src = lib_dir.join(real_name);
    if !real_src.exists() {
        return Err(format!("missing {real_name} in {}", lib_dir.display()));
    }
    let real_dst = out_dir.join(real_name);
    std::fs::copy(&real_src, &real_dst).map_err(|err| format!("copy {real_name}: {err}"))?;
    symlink_in_dir(real_name, &out_dir.join(compat_name))?;
    symlink_in_dir(compat_name, &out_dir.join(link_name))?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn symlink_in_dir(target_name: &str, dst: &PathBuf) -> Result<(), String> {
    if dst.exists() {
        std::fs::remove_file(dst).map_err(|err| format!("remove {}: {err}", dst.display()))?;
    }
    std::os::unix::fs::symlink(target_name, dst)
        .map_err(|err| format!("symlink {} -> {target_name}: {err}", dst.display()))
}

struct GrokConfig {
    lib_dir: PathBuf,
    source_include: PathBuf,
    build_include: PathBuf,
}

fn grok_config() -> Option<GrokConfig> {
    let source_root = std::env::var_os("SIGNINUM_GROK_SOURCE")
        .map_or_else(|| PathBuf::from("/tmp/grok-signinum"), PathBuf::from);
    let lib_dir = std::env::var_os("SIGNINUM_GROK_ROOT")
        .map_or_else(|| source_root.join("build/bin"), PathBuf::from);
    let source_include = source_root.join("src/lib/core");
    let build_include = source_root.join("build/src/lib/core");
    if has_grok_artifacts(&lib_dir, &source_include, &build_include) {
        Some(GrokConfig {
            lib_dir,
            source_include,
            build_include,
        })
    } else {
        None
    }
}

fn has_grok_artifacts(lib_dir: &Path, source_include: &Path, build_include: &Path) -> bool {
    let header = source_include.join("grok.h");
    let config_header = build_include.join("grk_config.h");
    if !(header.exists() && config_header.exists()) {
        return false;
    }
    [
        lib_dir.join("libgrokj2k.dylib"),
        lib_dir.join("libgrokj2k.so"),
        lib_dir.join("grokj2k.lib"),
    ]
    .into_iter()
    .any(|path| path.exists())
}
