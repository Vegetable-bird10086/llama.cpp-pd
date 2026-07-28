#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${1:-"${repo_root}/build-android"}
ndk_root=${ANDROID_NDK_ROOT:-/root/autodl-tmp/android-ndk-r27d}

if [[ ! -x "${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump" ]]; then
    echo "Android NDK was not found at ${ndk_root}" >&2
    exit 1
fi

cmake -S "${repo_root}" -B "${build_dir}" \
    -DCMAKE_TOOLCHAIN_FILE="${ndk_root}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-26 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8-a \
    -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF

cmake --build "${build_dir}" -j"$(nproc)" \
    --target llama-pd-cli qnn-u16-core-debug

objdump="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump"
cpu_library="${build_dir}/bin/libggml-cpu.so"
if ! "${objdump}" -d "${cpu_library}" |
        grep -E '[[:space:]]sdot[[:space:]]' >/dev/null; then
    echo "SDOT instructions are missing from ${cpu_library}" >&2
    exit 1
fi

echo "Android PD build passed: portable ARMv8 baseline with runtime-gated SDOT"
