#!/usr/bin/env bash
set -euo pipefail

crate="${1:?usage: publish-crate.sh <crate>}"
dry_run="${DRY_RUN_ONLY:-false}"

if [[ "$dry_run" == "true" ]]; then
  cargo publish -p "$crate" --dry-run
  exit 0
fi

: "${CRATES_IO_API_TOKEN:?CRATES_IO_API_TOKEN is required for a real publish}"

cargo publish -p "$crate" --token "$CRATES_IO_API_TOKEN"
sleep "${CRATES_IO_INDEX_SETTLE_SECONDS:-30}"
