//! Authoritative benchmark packages, executables, lanes, features, and runtime gates.

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum BenchmarkLane {
    Host,
    Cuda,
    Metal,
    All,
}

impl BenchmarkLane {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "host" => Ok(Self::Host),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "all" => Ok(Self::All),
            _ => Err(format!(
                "unknown benchmark lane `{value}`; expected host, cuda, metal, or all"
            )),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::All => "all",
        }
    }

    pub(crate) fn selects(self, lane: Self) -> bool {
        self == Self::All || self == lane
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompileBenchmark {
    pub(crate) package: &'static str,
    pub(crate) bench: Option<&'static str>,
    pub(crate) features: Option<&'static str>,
    pub(crate) lane: BenchmarkLane,
    pub(crate) runtime_env: &'static [(&'static str, &'static str)],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PerformanceBenchmark {
    pub(crate) package: &'static str,
    pub(crate) bench: &'static str,
    pub(crate) filter: Option<&'static str>,
    pub(crate) features: Option<&'static str>,
    pub(crate) lane: BenchmarkLane,
    pub(crate) env: &'static [(&'static str, &'static str)],
}

pub(crate) const CUDA_BENCH_ENV: &[(&str, &str)] = &[
    ("J2K_REQUIRE_CUDA_BENCH", "1"),
    ("J2K_REQUIRE_CUDA_RUNTIME", "1"),
];
pub(crate) const METAL_BENCH_ENV: &[(&str, &str)] = &[
    ("J2K_REQUIRE_METAL_BENCH", "1"),
    ("J2K_REQUIRE_METAL_RUNTIME", "1"),
];

const fn compile(
    package: &'static str,
    bench: Option<&'static str>,
    features: Option<&'static str>,
    lane: BenchmarkLane,
    runtime_env: &'static [(&'static str, &'static str)],
) -> CompileBenchmark {
    CompileBenchmark {
        package,
        bench,
        features,
        lane,
        runtime_env,
    }
}

const fn performance(
    package: &'static str,
    bench: &'static str,
    filter: Option<&'static str>,
    features: Option<&'static str>,
    lane: BenchmarkLane,
    env: &'static [(&'static str, &'static str)],
) -> PerformanceBenchmark {
    PerformanceBenchmark {
        package,
        bench,
        filter,
        features,
        lane,
        env,
    }
}

pub(crate) const COMPILE_BENCHMARKS: &[CompileBenchmark] = &[
    compile("j2k", Some("public_api"), None, BenchmarkLane::Host, &[]),
    compile(
        "j2k-native",
        Some("tier1_bitplane"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    compile(
        "j2k-native",
        Some("htj2k_sigprop_phase"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    compile(
        "j2k-native",
        Some("direct_cpu"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    compile(
        "j2k-jpeg",
        Some("encode_cpu"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    compile(
        "j2k-jpeg",
        None,
        Some("bench-libjpeg-turbo"),
        BenchmarkLane::Host,
        &[],
    ),
    compile(
        "j2k-tilecodec",
        Some("compare"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    compile(
        "j2k-transcode",
        Some("dct53"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    compile(
        "j2k-ml",
        Some("batch_decode"),
        Some("cpu"),
        BenchmarkLane::Host,
        &[],
    ),
    compile(
        "j2k-jpeg-cuda",
        Some("device_decode"),
        Some("cuda-runtime"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    compile(
        "j2k-jpeg-cuda",
        Some("decode_defusion"),
        Some("cuda-runtime"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    compile(
        "j2k-cuda",
        Some("auto_routing"),
        Some("cuda-runtime"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    compile(
        "j2k-cuda",
        Some("encode_stages"),
        Some("cuda-runtime"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    compile(
        "j2k-cuda",
        Some("htj2k_decode"),
        Some("cuda-runtime"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    compile(
        "j2k-cuda",
        Some("htj2k_encode"),
        Some("cuda-runtime"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    compile(
        "j2k-ml",
        Some("batch_decode_cuda"),
        Some("cpu,cuda"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    compile(
        "j2k-metal",
        Some("auto_routing"),
        None,
        BenchmarkLane::Metal,
        METAL_BENCH_ENV,
    ),
    compile(
        "j2k-jpeg-metal",
        None,
        None,
        BenchmarkLane::Metal,
        METAL_BENCH_ENV,
    ),
    compile(
        "j2k-transcode-metal",
        Some("dct97"),
        Some("bench-internals"),
        BenchmarkLane::Metal,
        METAL_BENCH_ENV,
    ),
    compile(
        "j2k-ml",
        Some("batch_decode_metal"),
        Some("cpu,metal"),
        BenchmarkLane::Metal,
        METAL_BENCH_ENV,
    ),
];

pub(crate) const PERFORMANCE_BENCHMARKS: &[PerformanceBenchmark] = &[
    performance("j2k", "public_api", None, None, BenchmarkLane::Host, &[]),
    performance(
        "j2k-jpeg",
        "encode_cpu",
        Some("jpeg_cpu_encode_runtime/"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    performance(
        "j2k-native",
        "tier1_bitplane",
        Some("htj2k_cleanup_decode/"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    performance(
        "j2k-native",
        "tier1_bitplane",
        Some("htj2k_refinement_fixture_decode"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    performance(
        "j2k-native",
        "tier1_bitplane",
        Some("htj2k_refinement_block_decode"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    performance(
        "j2k-native",
        "tier1_bitplane",
        Some("htj2k_cleanup_encode/"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    performance(
        "j2k-native",
        "tier1_bitplane",
        Some("htj2k_cleanup_encode_distribution"),
        None,
        BenchmarkLane::Host,
        &[],
    ),
    performance(
        "j2k-native",
        "htj2k_sigprop_phase",
        None,
        None,
        BenchmarkLane::Host,
        &[],
    ),
    performance(
        "j2k-cuda",
        "htj2k_decode",
        Some("j2k_cuda_htj2k_"),
        Some("cuda-runtime"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    performance(
        "j2k-cuda",
        "htj2k_encode",
        Some("j2k_cuda_htj2k_"),
        Some("cuda-runtime"),
        BenchmarkLane::Cuda,
        CUDA_BENCH_ENV,
    ),
    performance(
        "j2k-jpeg-metal",
        "compare",
        None,
        None,
        BenchmarkLane::Metal,
        METAL_BENCH_ENV,
    ),
];

#[cfg(test)]
mod tests {
    use super::{compile, performance, BenchmarkLane};

    #[test]
    fn registry_helpers_preserve_lane_names_and_benchmark_fields() {
        for (lane, name) in [
            (BenchmarkLane::Host, "host"),
            (BenchmarkLane::Cuda, "cuda"),
            (BenchmarkLane::Metal, "metal"),
            (BenchmarkLane::All, "all"),
        ] {
            assert_eq!(lane.as_str(), name);
        }

        let compile_benchmark = compile(
            "compile-package",
            Some("compile-bench"),
            Some("compile-feature"),
            BenchmarkLane::Cuda,
            &[("COMPILE_ENV", "1")],
        );
        assert_eq!(compile_benchmark.package, "compile-package");
        assert_eq!(compile_benchmark.bench, Some("compile-bench"));
        assert_eq!(compile_benchmark.features, Some("compile-feature"));
        assert_eq!(compile_benchmark.lane, BenchmarkLane::Cuda);
        assert_eq!(compile_benchmark.runtime_env, &[("COMPILE_ENV", "1")]);

        let performance_benchmark = performance(
            "performance-package",
            "performance-bench",
            Some("performance-filter"),
            Some("performance-feature"),
            BenchmarkLane::Metal,
            &[("PERFORMANCE_ENV", "1")],
        );
        assert_eq!(performance_benchmark.package, "performance-package");
        assert_eq!(performance_benchmark.bench, "performance-bench");
        assert_eq!(performance_benchmark.filter, Some("performance-filter"));
        assert_eq!(performance_benchmark.features, Some("performance-feature"));
        assert_eq!(performance_benchmark.lane, BenchmarkLane::Metal);
        assert_eq!(performance_benchmark.env, &[("PERFORMANCE_ENV", "1")]);
    }

    #[test]
    fn auto_routing_benchmarks_are_compiled_on_their_hardware_lanes() {
        let cuda = super::COMPILE_BENCHMARKS
            .iter()
            .find(|benchmark| {
                benchmark.package == "j2k-cuda" && benchmark.bench == Some("auto_routing")
            })
            .expect("CUDA Auto-routing benchmark registry entry");
        assert_eq!(cuda.features, Some("cuda-runtime"));
        assert_eq!(cuda.lane, BenchmarkLane::Cuda);
        assert_eq!(cuda.runtime_env, super::CUDA_BENCH_ENV);

        let metal = super::COMPILE_BENCHMARKS
            .iter()
            .find(|benchmark| {
                benchmark.package == "j2k-metal" && benchmark.bench == Some("auto_routing")
            })
            .expect("Metal Auto-routing benchmark registry entry");
        assert_eq!(metal.features, None);
        assert_eq!(metal.lane, BenchmarkLane::Metal);
        assert_eq!(metal.runtime_env, super::METAL_BENCH_ENV);
    }

    #[test]
    fn gpu_benchmark_workflow_verifies_both_auto_routing_lanes() {
        let workflow = include_str!("../../.github/workflows/gpu-benchmarks.yml");
        assert!(workflow.contains("options: [smoke, criterion, profile, adoption, routing]"));
        let workflow = include_str!("../../.github/workflows/gpu-benchmarks-runner.yml");
        assert!(workflow.contains(
            "cargo bench --profile release-bench -p j2k-cuda --bench auto_routing --features cuda-runtime"
        ));
        assert!(workflow
            .contains("cargo bench --profile release-bench -p j2k-metal --bench auto_routing"));
        assert_eq!(
            workflow.matches("cargo xtask auto-routing verify").count(),
            2
        );
    }
}
