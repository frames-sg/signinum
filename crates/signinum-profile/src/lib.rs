//! Internal profiling helpers shared by the `signinum` workspace crates.

#![doc(hidden)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

/// Controls profiling output for a profiling stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileStageMode {
    /// Disable profiling output.
    Disabled,
    /// Emit one row per profiling event.
    Rows,
    /// Aggregate profiling events and emit summary rows.
    Summary,
}

/// Maps an input profiling field to the label name used in summaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SummaryLabel {
    input_key: String,
    summary_key: String,
}

impl SummaryLabel {
    /// Creates a label remapping from an input field key to a summary key.
    pub fn new(input_key: impl AsRef<str>, summary_key: impl AsRef<str>) -> Self {
        Self {
            input_key: input_key.as_ref().to_owned(),
            summary_key: summary_key.as_ref().to_owned(),
        }
    }

    /// Creates a label whose input and summary keys are the same.
    pub fn same(key: impl AsRef<str>) -> Self {
        Self::new(key.as_ref(), key.as_ref())
    }
}

/// Returns whether an optional environment value is a recognized truthy flag.
pub fn env_flag_from_value(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };
    let value = value.trim();

    matches!(value, "1")
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("t")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("y")
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("enable")
        || value.eq_ignore_ascii_case("enabled")
}

/// Parses a profiling stage mode from an optional environment value.
pub fn profile_stage_mode_from_value(value: Option<&str>) -> ProfileStageMode {
    let Some(value) = value else {
        return ProfileStageMode::Disabled;
    };
    let value = value.trim();

    if value.eq_ignore_ascii_case("summary")
        || value.eq_ignore_ascii_case("summaries")
        || value.eq_ignore_ascii_case("aggregate")
        || value.eq_ignore_ascii_case("aggregates")
    {
        ProfileStageMode::Summary
    } else if env_flag_from_value(Some(value)) {
        ProfileStageMode::Rows
    } else {
        ProfileStageMode::Disabled
    }
}

/// Formats a profiling row from string fields.
pub fn format_profile_row<K, V>(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, V)],
) -> String
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut row = format_profile_prefix(codec.as_ref(), op.as_ref(), path.as_ref());
    for (key, value) in fields {
        write!(row, " {}={}", key.as_ref(), value.as_ref()).expect("writing to String failed");
    }
    row
}

/// Formats a profiling row from integer fields.
pub fn format_profile_row_u128<K>(
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, u128)],
) -> String
where
    K: AsRef<str>,
{
    let mut row = format_profile_prefix(codec.as_ref(), op.as_ref(), path.as_ref());
    for (key, value) in fields {
        write!(row, " {}={value}", key.as_ref()).expect("writing to String failed");
    }
    row
}

fn format_profile_prefix(codec: &str, op: &str, path: &str) -> String {
    let mut row = String::new();
    write!(row, "signinum_profile codec={codec} op={op} path={path}")
        .expect("writing to String failed");
    row
}

/// Aggregates profiling rows by codec, operation, path, and configured labels.
#[derive(Clone, Debug)]
pub struct ProfileSummary {
    labels: Vec<SummaryLabel>,
    numeric_mode: SummaryNumericMode,
    rows: BTreeMap<SummaryKey, SummaryRow>,
}

impl ProfileSummary {
    /// Creates an empty profile summary with the given summary labels.
    pub fn new(labels: impl IntoIterator<Item = SummaryLabel>) -> Self {
        Self::with_numeric_mode(labels, SummaryNumericMode::Aggregate)
    }

    /// Creates an empty profile summary that counts rows without aggregating numeric fields.
    pub fn counts_only(labels: impl IntoIterator<Item = SummaryLabel>) -> Self {
        Self::with_numeric_mode(labels, SummaryNumericMode::CountOnly)
    }

    fn with_numeric_mode(
        labels: impl IntoIterator<Item = SummaryLabel>,
        numeric_mode: SummaryNumericMode,
    ) -> Self {
        Self {
            labels: labels.into_iter().collect(),
            numeric_mode,
            rows: BTreeMap::new(),
        }
    }

    /// Records a profiling row with string field values.
    pub fn record_str<K, V>(
        &mut self,
        codec: impl AsRef<str>,
        op: impl AsRef<str>,
        path: impl AsRef<str>,
        fields: &[(K, V)],
    ) where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let key = self.summary_key_from_str(codec.as_ref(), op.as_ref(), path.as_ref(), fields);
        let mut numeric_fields = Vec::new();
        if self.numeric_mode == SummaryNumericMode::Aggregate {
            for (field_key, field_value) in fields {
                let field_key = field_key.as_ref();
                if self.is_label_key(field_key) {
                    continue;
                }
                if let Ok(value) = field_value.as_ref().parse::<u128>() {
                    numeric_fields.push((field_key.to_owned(), value));
                }
            }
        }

        let row = self.rows.entry(key).or_default();
        row.count = row.count.saturating_add(1);
        for (field_key, value) in numeric_fields {
            row.record_numeric(&field_key, value);
        }
    }

    /// Records a profiling row with unsigned integer field values.
    pub fn record_u128<K>(
        &mut self,
        codec: impl AsRef<str>,
        op: impl AsRef<str>,
        path: impl AsRef<str>,
        fields: &[(K, u128)],
    ) where
        K: AsRef<str>,
    {
        let key = self.summary_key_from_u128(codec.as_ref(), op.as_ref(), path.as_ref(), fields);
        let mut numeric_fields = Vec::new();
        if self.numeric_mode == SummaryNumericMode::Aggregate {
            for (field_key, value) in fields {
                let field_key = field_key.as_ref();
                if self.is_label_key(field_key) {
                    continue;
                }
                numeric_fields.push((field_key.to_owned(), *value));
            }
        }

        let row = self.rows.entry(key).or_default();
        row.count = row.count.saturating_add(1);
        for (field_key, value) in numeric_fields {
            row.record_numeric(&field_key, value);
        }
    }

    /// Formats deterministic summary rows.
    pub fn format_rows(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|(key, row)| row.format_with_key(key))
            .collect()
    }

    fn summary_key_from_str<K, V>(
        &self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &[(K, V)],
    ) -> SummaryKey
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let labels = self
            .labels
            .iter()
            .filter_map(|label| {
                find_str_field(fields, &label.input_key)
                    .map(|field_value| (label.summary_key.clone(), field_value))
            })
            .collect();

        SummaryKey::new(codec, op, path, labels)
    }

    fn summary_key_from_u128<K>(
        &self,
        codec: &str,
        op: &str,
        path: &str,
        fields: &[(K, u128)],
    ) -> SummaryKey
    where
        K: AsRef<str>,
    {
        let labels = self
            .labels
            .iter()
            .filter_map(|label| {
                find_u128_field(fields, &label.input_key)
                    .map(|field_value| (label.summary_key.clone(), field_value))
            })
            .collect();

        SummaryKey::new(codec, op, path, labels)
    }

    fn is_label_key(&self, key: &str) -> bool {
        self.labels.iter().any(|label| label.input_key == key)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SummaryNumericMode {
    Aggregate,
    CountOnly,
}

impl Default for ProfileSummary {
    fn default() -> Self {
        Self::new([])
    }
}

#[cfg(feature = "std")]
impl Drop for ProfileSummary {
    fn drop(&mut self) {
        for row in self.format_rows() {
            std::eprintln!("{row}");
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SummaryKey {
    codec: String,
    op: String,
    path: String,
    labels: Vec<(String, String)>,
}

impl SummaryKey {
    fn new(codec: &str, op: &str, path: &str, labels: Vec<(String, String)>) -> Self {
        Self {
            codec: codec.to_owned(),
            op: op.to_owned(),
            path: path.to_owned(),
            labels,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct SummaryRow {
    count: u128,
    numeric_sums: BTreeMap<String, u128>,
}

impl SummaryRow {
    fn record_numeric(&mut self, key: &str, value: u128) {
        self.numeric_sums
            .entry(key.to_owned())
            .and_modify(|sum| *sum = sum.saturating_add(value))
            .or_insert(value);
    }

    fn format_with_key(&self, key: &SummaryKey) -> String {
        let mut row = format_profile_summary_prefix(&key.codec, &key.op, &key.path);

        for (label_key, label_value) in &key.labels {
            write!(row, " {label_key}={label_value}").expect("writing to String failed");
        }
        write!(row, " count={}", self.count).expect("writing to String failed");

        for (field_key, sum) in &self.numeric_sums {
            write!(row, " {field_key}_sum={sum}").expect("writing to String failed");
            if is_timing_field(field_key) {
                let average = sum / self.count;
                write!(row, " {field_key}_avg={average}").expect("writing to String failed");
            }
        }

        row
    }
}

fn format_profile_summary_prefix(codec: &str, op: &str, path: &str) -> String {
    let mut row = String::new();
    write!(
        row,
        "signinum_profile_summary codec={codec} op={op} path={path}"
    )
    .expect("writing to String failed");
    row
}

fn find_str_field<K, V>(fields: &[(K, V)], key: &str) -> Option<String>
where
    K: AsRef<str>,
    V: AsRef<str>,
{
    fields
        .iter()
        .find(|(field_key, _)| field_key.as_ref() == key)
        .map(|(_, field_value)| field_value.as_ref().to_owned())
}

fn find_u128_field<K>(fields: &[(K, u128)], key: &str) -> Option<String>
where
    K: AsRef<str>,
{
    fields
        .iter()
        .find(|(field_key, _)| field_key.as_ref() == key)
        .map(|(_, field_value)| {
            let mut value = String::new();
            write!(value, "{field_value}").expect("writing to String failed");
            value
        })
}

fn is_timing_field(field_key: &str) -> bool {
    field_key.ends_with("_us") || field_key.ends_with("_ms") || field_key.ends_with("_ns")
}

#[cfg(feature = "std")]
/// Emits or records a string-valued profiling row according to the stage mode.
pub fn emit_profile_row<K, V>(
    mode: ProfileStageMode,
    summary: &'static std::thread::LocalKey<std::cell::RefCell<ProfileSummary>>,
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, V)],
) where
    K: AsRef<str>,
    V: AsRef<str>,
{
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            std::eprintln!("{}", format_profile_row(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                summary
                    .borrow_mut()
                    .record_str(codec.as_ref(), op.as_ref(), path.as_ref(), fields);
            });
        }
    }
}

#[cfg(feature = "std")]
/// Emits or records an integer-valued profiling row according to the stage mode.
pub fn emit_profile_row_u128<K>(
    mode: ProfileStageMode,
    summary: &'static std::thread::LocalKey<std::cell::RefCell<ProfileSummary>>,
    codec: impl AsRef<str>,
    op: impl AsRef<str>,
    path: impl AsRef<str>,
    fields: &[(K, u128)],
) where
    K: AsRef<str>,
{
    match mode {
        ProfileStageMode::Disabled => {}
        ProfileStageMode::Rows => {
            std::eprintln!("{}", format_profile_row_u128(codec, op, path, fields));
        }
        ProfileStageMode::Summary => {
            summary.with(|summary| {
                summary.borrow_mut().record_u128(
                    codec.as_ref(),
                    op.as_ref(),
                    path.as_ref(),
                    fields,
                );
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[cfg(feature = "std")]
    use std::cell::RefCell;

    #[test]
    fn parses_env_truthy_and_falsy_values() {
        for value in [
            Some("1"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("on"),
        ] {
            assert!(env_flag_from_value(value));
        }

        for value in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("FALSE"),
            Some("no"),
            Some("off"),
        ] {
            assert!(!env_flag_from_value(value));
        }
    }

    #[test]
    fn parses_stage_mode_values() {
        assert_eq!(
            ProfileStageMode::Disabled,
            profile_stage_mode_from_value(None)
        );
        assert_eq!(
            ProfileStageMode::Disabled,
            profile_stage_mode_from_value(Some("off"))
        );
        assert_eq!(
            ProfileStageMode::Rows,
            profile_stage_mode_from_value(Some("1"))
        );
        assert_eq!(
            ProfileStageMode::Rows,
            profile_stage_mode_from_value(Some("true"))
        );

        for value in ["summary", "summaries", "aggregate", "aggregates"] {
            assert_eq!(
                ProfileStageMode::Summary,
                profile_stage_mode_from_value(Some(value))
            );
        }
    }

    #[test]
    fn formats_string_profile_rows() {
        let row = format_profile_row(
            "jpeg",
            "decode",
            "tile/0",
            &[("rows", "4"), ("elapsed_us", "12")],
        );

        assert_eq!(
            "signinum_profile codec=jpeg op=decode path=tile/0 rows=4 elapsed_us=12",
            row
        );
    }

    #[test]
    fn formats_u128_profile_rows() {
        let row = format_profile_row_u128(
            "j2k",
            "decode",
            "tile/1",
            &[("elapsed_us", 34_u128), ("bytes", 99_u128)],
        );

        assert_eq!(
            "signinum_profile codec=j2k op=decode path=tile/1 elapsed_us=34 bytes=99",
            row
        );
    }

    #[test]
    fn summary_counts_and_remaps_labels() {
        let mut summary = ProfileSummary::new(vec![
            SummaryLabel::new("component", "stage"),
            SummaryLabel::same("backend"),
        ]);

        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[
                ("component", "idct"),
                ("backend", "cpu"),
                ("quality", "fast"),
            ],
        );
        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("component", "idct"), ("backend", "cpu")],
        );

        assert_eq!(
            vec![
                "signinum_profile_summary codec=jpeg op=decode path=tile/0 stage=idct backend=cpu count=2"
            ],
            summary.format_rows()
        );
    }

    #[test]
    fn summary_emits_timing_sums_and_averages() {
        let mut summary = ProfileSummary::new([SummaryLabel::same("stage")]);

        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "entropy"), ("elapsed_us", "10"), ("bytes", "100")],
        );
        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "entropy"), ("elapsed_us", "20"), ("bytes", "50")],
        );

        assert_eq!(
            vec![
                "signinum_profile_summary codec=jpeg op=decode path=tile/0 stage=entropy count=2 bytes_sum=150 elapsed_us_sum=30 elapsed_us_avg=15"
            ],
            summary.format_rows()
        );
    }

    #[test]
    fn count_only_summary_omits_numeric_fields() {
        let mut summary = ProfileSummary::counts_only([SummaryLabel::same("route")]);

        summary.record_u128(
            "j2k-cuda",
            "decode",
            "tile/2",
            &[
                ("route", 1_u128),
                ("width", 512_u128),
                ("height", 512_u128),
                ("tiles", 4_u128),
            ],
        );
        summary.record_u128(
            "j2k-cuda",
            "decode",
            "tile/2",
            &[
                ("route", 1_u128),
                ("width", 256_u128),
                ("height", 256_u128),
                ("tiles", 2_u128),
            ],
        );

        assert_eq!(
            vec!["signinum_profile_summary codec=j2k-cuda op=decode path=tile/2 route=1 count=2"],
            summary.format_rows()
        );
    }

    #[test]
    fn summary_omits_absent_configured_labels() {
        let mut summary = ProfileSummary::new([
            SummaryLabel::same("backend"),
            SummaryLabel::same("missing_label"),
        ]);

        summary.record_str(
            "jpeg",
            "decode",
            "tile/0",
            &[("backend", "cpu"), ("elapsed_us", "8")],
        );

        assert_eq!(
            vec![
                "signinum_profile_summary codec=jpeg op=decode path=tile/0 backend=cpu count=1 elapsed_us_sum=8 elapsed_us_avg=8"
            ],
            summary.format_rows()
        );
    }

    #[test]
    fn summary_emits_u128_timing_summaries() {
        let mut summary = ProfileSummary::new([SummaryLabel::same("backend")]);

        summary.record_u128(
            "j2k",
            "decode",
            "tile/1",
            &[("backend", 7_u128), ("elapsed_ns", 3_u128)],
        );
        summary.record_u128(
            "j2k",
            "decode",
            "tile/1",
            &[("backend", 7_u128), ("elapsed_ns", 9_u128)],
        );

        assert_eq!(
            vec![
                "signinum_profile_summary codec=j2k op=decode path=tile/1 backend=7 count=2 elapsed_ns_sum=12 elapsed_ns_avg=6"
            ],
            summary.format_rows()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn profile_summary_drop_formats_rows_without_panic() {
        {
            let mut summary = ProfileSummary::new([SummaryLabel::same("stage")]);
            summary.record_str(
                "jpeg",
                "decode",
                "tile/0",
                &[("stage", "emit"), ("elapsed_ms", "4")],
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn emit_helpers_honor_stage_modes() {
        thread_local! {
            static TEST_SUMMARY: RefCell<ProfileSummary> =
                RefCell::new(ProfileSummary::new([SummaryLabel::same("stage")]));
        }

        emit_profile_row(
            ProfileStageMode::Disabled,
            &TEST_SUMMARY,
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "off"), ("elapsed_us", "10")],
        );
        TEST_SUMMARY.with(|summary| assert!(summary.borrow().format_rows().is_empty()));

        emit_profile_row(
            ProfileStageMode::Summary,
            &TEST_SUMMARY,
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", "on"), ("elapsed_us", "10")],
        );
        emit_profile_row_u128(
            ProfileStageMode::Summary,
            &TEST_SUMMARY,
            "jpeg",
            "decode",
            "tile/0",
            &[("stage", 1_u128), ("elapsed_us", 5_u128)],
        );

        TEST_SUMMARY.with(|summary| {
            assert_eq!(
                vec![
                    "signinum_profile_summary codec=jpeg op=decode path=tile/0 stage=1 count=1 elapsed_us_sum=5 elapsed_us_avg=5",
                    "signinum_profile_summary codec=jpeg op=decode path=tile/0 stage=on count=1 elapsed_us_sum=10 elapsed_us_avg=10",
                ],
                summary.borrow().format_rows()
            );
        });
    }
}
