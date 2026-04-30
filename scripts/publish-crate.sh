#!/usr/bin/env bash
set -euo pipefail

crate="${1:?usage: publish-crate.sh <crate>}"
dry_run="${DRY_RUN_ONLY:-false}"
version="$(cargo pkgid -p "$crate" | sed 's/.*#//')"

if [[ "$dry_run" == "true" ]]; then
  cargo publish -p "$crate" --dry-run
  exit 0
fi

: "${CRATES_IO_API_TOKEN:?CRATES_IO_API_TOKEN is required for a real publish}"

if cargo info "${crate}@${version}" --registry crates-io >/dev/null 2>&1; then
  echo "${crate} ${version} is already published; skipping"
  exit 0
fi

export CARGO_REGISTRY_TOKEN="$CRATES_IO_API_TOKEN"
cargo publish -p "$crate"
sleep "${CRATES_IO_INDEX_SETTLE_SECONDS:-30}"
