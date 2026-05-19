use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_grok)");
    println!("cargo:rerun-if-env-changed=SIGNINUM_GROK_ROOT");
    println!("cargo:rerun-if-env-changed=SIGNINUM_GROK_SOURCE");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
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
            config.lib_dir.display()
        );
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
        stage_grok_dylib_family(lib_dir, &out_dir)?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = lib_dir;
    }
    Ok(out_dir)
}

#[cfg(target_os = "macos")]
fn stage_grok_dylib_family(lib_dir: &Path, out_dir: &Path) -> Result<(), String> {
    let real_src = find_grok_real_dylib(lib_dir)?;
    let real_name = real_src
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| format!("invalid Grok dylib name: {}", real_src.display()))?;
    let real_dst = out_dir.join(real_name);
    if real_dst.exists() {
        std::fs::remove_file(&real_dst)
            .map_err(|err| format!("remove {}: {err}", real_dst.display()))?;
    }
    std::fs::copy(&real_src, &real_dst)
        .map_err(|err| format!("copy {}: {err}", real_src.display()))?;
    symlink_in_dir(real_name, &out_dir.join("libgrokj2k.1.dylib"))?;
    symlink_in_dir("libgrokj2k.1.dylib", &out_dir.join("libgrokj2k.dylib"))?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn find_grok_real_dylib(lib_dir: &Path) -> Result<PathBuf, String> {
    let mut candidates = std::fs::read_dir(lib_dir)
        .map_err(|err| format!("read Grok lib dir {}: {err}", lib_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("read Grok lib dir entry: {err}"))?;
    candidates.retain(|path| {
        path.file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| {
                name.starts_with("libgrokj2k.")
                    && path
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("dylib"))
                    && name != "libgrokj2k.dylib"
                    && name != "libgrokj2k.1.dylib"
                    && !name.contains("codec")
            })
    });
    candidates.sort();
    candidates.pop().ok_or_else(|| {
        format!(
            "missing versioned libgrokj2k dylib in {}",
            lib_dir.display()
        )
    })
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
        pkg_config_grok_config()
    }
}

fn pkg_config_grok_config() -> Option<GrokConfig> {
    let include_dir = pkg_config_variable("libgrokj2k", "includedir")?;
    let lib_dir = pkg_config_variable("libgrokj2k", "libdir")?;
    if has_grok_artifacts(&lib_dir, &include_dir, &include_dir) {
        Some(GrokConfig {
            lib_dir,
            source_include: include_dir.clone(),
            build_include: include_dir,
        })
    } else {
        None
    }
}

fn pkg_config_variable(package: &str, variable: &str) -> Option<PathBuf> {
    let output = Command::new("pkg-config")
        .args(["--variable", variable, package])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value))
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
