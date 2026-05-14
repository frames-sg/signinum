// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Duration;

use signinum_profile::{profile_stage_mode_from_value, ProfileStageMode, SummaryLabel};

#[cfg(test)]
pub(crate) use signinum_profile::{env_flag_from_value, format_profile_row, ProfileSummary};

const JPEG_PROFILE_STAGES_ENV: &str = "SIGNINUM_JPEG_PROFILE_STAGES";
const SUMMARY_LABEL_FIELD_KEYS: &[&str] = &["mode", "fmt", "downscale", "scan_path"];

pub(crate) fn jpeg_profile_stages_enabled() -> bool {
    jpeg_profile_stage_mode() != ProfileStageMode::Disabled
}

fn jpeg_profile_stage_mode() -> ProfileStageMode {
    static MODE: OnceLock<ProfileStageMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        profile_stage_mode_from_value(std::env::var(JPEG_PROFILE_STAGES_ENV).ok().as_deref())
    })
}

pub(crate) fn emit_jpeg_profile_row(op: &str, path: &str, fields: &[(&str, &str)]) {
    match jpeg_profile_stage_mode() {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            signinum_profile::emit_profile_row(
                ProfileStageMode::Rows,
                &PROFILE_SUMMARY,
                "jpeg",
                op,
                path,
                fields,
            );
        }
        ProfileStageMode::Summary => {
            PROFILE_SUMMARY.with(|summary| {
                record_jpeg_profile_summary(&mut summary.borrow_mut(), op, path, fields);
            });
        }
    }
}

pub(crate) fn duration_us_string(duration: Duration) -> String {
    duration.as_micros().to_string()
}

thread_local! {
    static PROFILE_SUMMARY: RefCell<signinum_profile::ProfileSummary> = RefCell::new(
        signinum_profile::ProfileSummary::new(summary_label_fields().iter().cloned())
    );
}

fn summary_label_fields() -> &'static [SummaryLabel] {
    static SUMMARY_LABEL_FIELDS: OnceLock<Box<[SummaryLabel]>> = OnceLock::new();
    SUMMARY_LABEL_FIELDS.get_or_init(|| {
        SUMMARY_LABEL_FIELD_KEYS
            .iter()
            .map(SummaryLabel::same)
            .collect()
    })
}

fn record_jpeg_profile_summary(
    summary: &mut signinum_profile::ProfileSummary,
    op: &str,
    path: &str,
    fields: &[(&str, &str)],
) {
    let summary_fields = fields
        .iter()
        .copied()
        .filter(|(field, _)| is_summary_label_field(field) || is_timing_field(field))
        .collect::<Vec<_>>();
    summary.record_str("jpeg", op, path, &summary_fields);
}

fn is_summary_label_field(field: &str) -> bool {
    SUMMARY_LABEL_FIELD_KEYS.contains(&field)
}

fn is_timing_field(field: &str) -> bool {
    field.ends_with("_us") || field.ends_with("_ns") || field.ends_with("_ms")
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
    fn env_flag_rejects_empty_false_and_zero_values() {
        for value in ["", "0", "false", "FALSE", "no", "off", "anything-else"] {
            assert!(
                !env_flag_from_value(Some(value)),
                "{value} should disable profiling"
            );
        }
        assert!(!env_flag_from_value(None));
    }

    #[test]
    fn profile_row_uses_compact_key_value_format() {
        let fields = [("width", "19"), ("height", "17"), ("total_us", "123")];
        let row = format_profile_row("jpeg", "encode", "cpu", &fields);
        assert_eq!(
            row,
            "signinum_profile codec=jpeg op=encode path=cpu width=19 height=17 total_us=123"
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
    fn profile_summary_groups_rows_and_averages_timing_fields() {
        let mut summary = ProfileSummary::new(summary_label_fields().iter().cloned());
        record_jpeg_profile_summary(
            &mut summary,
            "decode",
            "cpu",
            &[
                ("mode", "full"),
                ("fmt", "Rgb8"),
                ("source_width", "16"),
                ("decode_us", "4"),
                ("output_bytes", "48"),
                ("total_us", "6"),
            ],
        );
        record_jpeg_profile_summary(
            &mut summary,
            "decode",
            "cpu",
            &[
                ("mode", "full"),
                ("fmt", "Rgb8"),
                ("source_width", "32"),
                ("decode_us", "8"),
                ("output_bytes", "96"),
                ("total_us", "10"),
            ],
        );

        assert_eq!(
            summary.format_rows(),
            vec![
                "signinum_profile_summary codec=jpeg op=decode path=cpu mode=full fmt=Rgb8 count=2 decode_us_sum=12 decode_us_avg=6 total_us_sum=16 total_us_avg=8"
            ]
        );
    }
}
