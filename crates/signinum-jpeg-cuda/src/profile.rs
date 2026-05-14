// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::sync::OnceLock;

use signinum_profile::{profile_stage_mode_from_value, ProfileStageMode, SummaryLabel};

#[cfg(test)]
pub(crate) use signinum_profile::{env_flag_from_value, format_profile_row, ProfileSummary};

const GPU_ROUTE_PROFILE_ENV: &str = "SIGNINUM_GPU_ROUTE_PROFILE";

pub(crate) fn gpu_route_profile_enabled() -> bool {
    gpu_route_profile_stage_mode() != ProfileStageMode::Disabled
}

fn gpu_route_profile_stage_mode() -> ProfileStageMode {
    static MODE: OnceLock<ProfileStageMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        profile_stage_mode_from_value(std::env::var(GPU_ROUTE_PROFILE_ENV).ok().as_deref())
    })
}

pub(crate) fn emit_gpu_route_profile(codec: &str, op: &str, path: &str, fields: &[(&str, &str)]) {
    signinum_profile::emit_profile_row(
        gpu_route_profile_stage_mode(),
        &PROFILE_SUMMARY,
        codec,
        op,
        path,
        fields,
    );
}

thread_local! {
    static PROFILE_SUMMARY: RefCell<signinum_profile::ProfileSummary> = RefCell::new(
        signinum_profile::ProfileSummary::counts_only(summary_label_fields().iter().cloned())
    );
}

fn summary_label_fields() -> &'static [SummaryLabel] {
    static SUMMARY_LABEL_FIELDS: OnceLock<Box<[SummaryLabel]>> = OnceLock::new();
    SUMMARY_LABEL_FIELDS.get_or_init(|| {
        vec![
            SummaryLabel::new("op", "route_op"),
            SummaryLabel::same("request"),
            SummaryLabel::same("fmt"),
            SummaryLabel::same("decision"),
            SummaryLabel::same("reason"),
            SummaryLabel::same("has_fast_packet"),
            SummaryLabel::same("supports_output_format"),
            SummaryLabel::same("hardware_decode"),
        ]
        .into_boxed_slice()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_flag_accepts_common_truthy_values() {
        for value in ["1", "true", "TRUE", "yes", "on", "On"] {
            assert!(
                env_flag_from_value(Some(value)),
                "{value} should enable profiling"
            );
        }
    }

    #[test]
    fn profile_row_uses_compact_key_value_format() {
        let row = format_profile_row(
            "jpeg",
            "gpu_route",
            "cuda",
            &[("request", "Cuda"), ("decision", "nvjpeg")],
        );
        assert_eq!(
            row,
            "signinum_profile codec=jpeg op=gpu_route path=cuda request=Cuda decision=nvjpeg"
        );
    }

    #[test]
    fn profile_stage_mode_parses_summary_mode() {
        assert_eq!(
            profile_stage_mode_from_value(Some("summary")),
            ProfileStageMode::Summary
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("aggregate")),
            ProfileStageMode::Summary
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("1")),
            ProfileStageMode::Rows
        );
        assert_eq!(
            profile_stage_mode_from_value(Some("0")),
            ProfileStageMode::Disabled
        );
    }

    #[test]
    fn profile_summary_counts_route_decisions() {
        let mut summary = ProfileSummary::counts_only(summary_label_fields().iter().cloned());
        summary.record_str(
            "jpeg",
            "gpu_route",
            "cuda",
            &[
                ("op", "batch_full"),
                ("request", "AutoOrCuda"),
                ("fmt", "Rgb8"),
                ("tiles", "64"),
                ("decision", "nvjpeg_batch"),
            ],
        );
        summary.record_str(
            "jpeg",
            "gpu_route",
            "cuda",
            &[
                ("op", "batch_full"),
                ("request", "AutoOrCuda"),
                ("fmt", "Rgb8"),
                ("tiles", "128"),
                ("decision", "nvjpeg_batch"),
            ],
        );

        assert_eq!(
            summary.format_rows(),
            vec![
                "signinum_profile_summary codec=jpeg op=gpu_route path=cuda route_op=batch_full request=AutoOrCuda fmt=Rgb8 decision=nvjpeg_batch count=2"
            ]
        );
    }
}
