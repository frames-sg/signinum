#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
reference_common="reference-build-common.sh"
source "${script_dir}/${reference_common}"

version="0.19.0"
source_url="https://github.com/osamu620/OpenHTJ2K.git"
source_commit="e0f7ae853220d1e359c438b0bb6ad6cb2b3899db"
source_dir="${1:-target/t803/openhtj2k-v${version}}"
build_dir="${source_dir}/build-reference"

reference_prepare_checkout \
  "OpenHTJ2K" \
  "${source_dir}" \
  "${source_url}" \
  "v${version}" \
  "${source_commit}"

# OpenHTJ2K replaces the configuration flags that carry /MD under old CMake
# policy. Select the DLL runtime through the target property so its static
# library matches the Rust cc shim even after that replacement.
cmake \
  -S "${source_dir}" \
  -B "${build_dir}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL \
  -DOPENHTJ2K_QUIC=OFF
cmake \
  --build "${build_dir}" \
  --config Release \
  --target open_htj2k_dec \
  --parallel 2

decoder="$(reference_find_artifact \
  "OpenHTJ2K build did not produce open_htj2k_dec" \
  "${build_dir}/bin/open_htj2k_dec" \
  "${build_dir}/bin/open_htj2k_dec.exe" \
  "${build_dir}/bin/Release/open_htj2k_dec.exe")"

library="$(reference_find_artifact \
  "OpenHTJ2K build did not produce the static reference library" \
  "${build_dir}/libopenhtj2k.a" \
  "${build_dir}/openhtj2k.lib" \
  "${build_dir}/Release/openhtj2k.lib")"

source_dir="$(reference_canonical_dir "${source_dir}")"
decoder="$(reference_canonical_file "${decoder}")"
lib_dir="$(reference_canonical_dir "$(dirname "${library}")")"

reference_emit_env \
  "J2K_OPENHTJ2K_DEC_BIN=${decoder}" \
  "J2K_OPENHTJ2K_SOURCE_DIR=${source_dir}" \
  "J2K_OPENHTJ2K_LIB_DIR=${lib_dir}"
