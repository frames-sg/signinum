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
attempt=1
max_attempts="${CRATES_IO_PUBLISH_ATTEMPTS:-3}"
retry_seconds="${CRATES_IO_RATE_LIMIT_RETRY_SECONDS:-330}"

while true; do
  set +e
  output="$(cargo publish -p "$crate" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$output"

  if [[ "$status" -eq 0 ]]; then
    break
  fi

  if [[ "$output" != *"Too Many Requests"* || "$attempt" -ge "$max_attempts" ]]; then
    exit "$status"
  fi

  attempt=$((attempt + 1))
  echo "crates.io rate limited ${crate}; sleeping ${retry_seconds}s before retry ${attempt}/${max_attempts}"
  sleep "$retry_seconds"
done

sleep "${CRATES_IO_INDEX_SETTLE_SECONDS:-30}"
