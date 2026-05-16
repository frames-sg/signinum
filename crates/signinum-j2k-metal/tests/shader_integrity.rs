const COMPUTE_SOURCE: &str = include_str!("../src/compute.rs");
const SHADER_SOURCES: &[&str] = &[
    COMPUTE_SOURCE,
    include_str!("../src/classic.metal"),
    include_str!("../src/encode_bitstream.metal"),
    include_str!("../src/fdwt.metal"),
    include_str!("../src/ht_cleanup.metal"),
    include_str!("../src/idwt.metal"),
    include_str!("../src/mct.metal"),
    include_str!("../src/store.metal"),
];

#[test]
fn metal_kernels_are_wired_to_host_pipelines() {
    let unused = SHADER_SOURCES
        .iter()
        .flat_map(|source| kernel_names(source))
        .filter(|name| !host_compiles_pipeline(name))
        .collect::<Vec<_>>();

    assert!(
        unused.is_empty(),
        "Metal kernels must be compiled by host pipeline setup or removed: {unused:?}"
    );
}

fn kernel_names(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let rest = line.strip_prefix("kernel void ")?;
            let name = rest
                .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                .next()
                .expect("kernel name follows kernel declaration");
            Some(name.to_owned())
        })
        .collect()
}

fn host_compiles_pipeline(name: &str) -> bool {
    let quoted = format!("\"{name}\"");
    COMPUTE_SOURCE.match_indices(&quoted).any(|(index, _)| {
        let context_start = index.saturating_sub(96);
        let context = &COMPUTE_SOURCE[context_start..index];
        context.contains("get_function(") || context.contains("pipeline(")
    })
}
