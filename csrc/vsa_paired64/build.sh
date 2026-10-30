#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PACKAGE_DIR="${SCRIPT_DIR}/block_sparse_attention"
readonly BUILD_DIR="${SCRIPT_DIR}/build"
readonly DIST_DIR="${SCRIPT_DIR}/dist"
readonly NATIVE_LIBRARY="libblock_sparse_attention.so"

log() {
    printf '[block_sparse_attention] %s\n' "$*"
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [build|clean|help]

Commands:
  build  Build the wheel into dist/ and refresh the in-place native library.
         This is the default command.
  clean  Remove build outputs, generated native libraries, and Python caches.
  help   Show this message.

Environment variables:
  MAX_JOBS          Maximum number of parallel compile jobs (default: up to 8).
  CATLASS_DIR       Optional path to CATLASS; the CMake defaults are used if unset.
  ASCEND_HOME_PATH  CANN installation path; normally set by CANN set_env.sh.
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'error: required command not found: %s\n' "$1" >&2
        exit 1
    fi
}

check_build_environment() {
    require_command python3
    require_command cmake

    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        printf '%s\n' \
            'error: CANN environment is not configured; source CANN set_env.sh first' >&2
        exit 1
    fi

    if ! python3 -c 'import torch, torch_npu' >/dev/null 2>&1; then
        printf '%s\n' \
            'error: compatible torch and torch_npu packages are required' >&2
        exit 1
    fi
}

clean() {
    log "cleaning generated files"

    rm -rf -- \
        "${BUILD_DIR}" \
        "${DIST_DIR}" \
        "${SCRIPT_DIR}/.pytest_cache"

    find "${SCRIPT_DIR}" -maxdepth 1 -type d -name '*.egg-info' \
        -exec rm -rf -- {} +
    find "${SCRIPT_DIR}" -type d -name '__pycache__' -prune \
        -exec rm -rf -- {} +
    find "${SCRIPT_DIR}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
    find "${PACKAGE_DIR}" -maxdepth 1 -type f \
        \( -name '_C*.so' -o -name '_C*.pyd' \) -delete
    rm -f -- \
        "${PACKAGE_DIR}/lib/${NATIVE_LIBRARY}" \
        "${PACKAGE_DIR}/lib/${NATIVE_LIBRARY}.tmp"

    log "clean complete"
}

sync_inplace_library() {
    local source="${BUILD_DIR}/lib/${NATIVE_LIBRARY}"
    local destination_dir="${PACKAGE_DIR}/lib"
    local destination="${destination_dir}/${NATIVE_LIBRARY}"
    local temporary="${destination}.tmp"

    if [[ ! -f "${source}" ]]; then
        printf 'error: native library was not generated: %s\n' "${source}" >&2
        exit 1
    fi

    mkdir -p "${destination_dir}"
    cp -p -- "${source}" "${temporary}"
    mv -f -- "${temporary}" "${destination}"
}

build() {
    check_build_environment
    mkdir -p "${DIST_DIR}"

    log "building wheel"
    python3 -m pip wheel \
        "${SCRIPT_DIR}" \
        --no-build-isolation \
        --no-deps \
        --wheel-dir "${DIST_DIR}"

    sync_inplace_library

    local wheels=("${DIST_DIR}"/*.whl)
    if (( ${#wheels[@]} == 0 )) || [[ ! -f "${wheels[0]}" ]]; then
        printf 'error: build completed without producing a wheel in %s\n' \
            "${DIST_DIR}" >&2
        exit 1
    fi

    log "build complete"
    printf 'wheel: %s\n' "${wheels[@]}"
    printf 'native library: %s\n' "${PACKAGE_DIR}/lib/${NATIVE_LIBRARY}"
}

case "${1:-build}" in
    build)
        build
        ;;
    clean)
        clean
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        printf 'error: unknown command: %s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac
