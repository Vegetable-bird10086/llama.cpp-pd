# Android GPTQ2_32 Build Notes

This repository contains a custom `GGML_TYPE_GPTQ2_32` path for running
GPTQ-style 2-bit asymmetric weights exported as GGUF.

For Android deployment, use the no-OpenMP build. The initial Android build
that linked against `libomp.so` is not suitable for a minimal phone-side
deployment because the runtime library must also be shipped.

## Configure

```bash
cmake -S /root/autodl-tmp/llama.cpp \
  -B /root/autodl-tmp/llama.cpp/build-android \
  -DCMAKE_TOOLCHAIN_FILE=/root/autodl-tmp/android-ndk-r27d/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_OPENMP=OFF
```

## Build

```bash
cmake --build /root/autodl-tmp/llama.cpp/build-android -j 8 --target llama-completion
```

## Push To Phone

```bash
adb shell rm -rf /data/local/tmp/llama-pd-noomp
adb shell mkdir -p /data/local/tmp/llama-pd-noomp

adb push /root/autodl-tmp/llama.cpp/build-android/bin/llama-completion /data/local/tmp/llama-pd-noomp/
adb push /root/autodl-tmp/llama.cpp/build-android/bin/libggml-base.so /data/local/tmp/llama-pd-noomp/
adb push /root/autodl-tmp/llama.cpp/build-android/bin/libggml-cpu.so /data/local/tmp/llama-pd-noomp/
adb push /root/autodl-tmp/llama.cpp/build-android/bin/libggml.so /data/local/tmp/llama-pd-noomp/
adb push /root/autodl-tmp/llama.cpp/build-android/bin/libllama.so /data/local/tmp/llama-pd-noomp/
adb push /root/autodl-tmp/llama.cpp/build-android/bin/libllama-common.so /data/local/tmp/llama-pd-noomp/
adb push /root/autodl-tmp/llama.cpp/build-android/bin/libllama-completion-impl.so /data/local/tmp/llama-pd-noomp/

adb push /root/autodl-tmp/Qwen3-1.7b-2bit/Qwen3-1.7b-2bit.gptq2_32.gguf /data/local/tmp/llama-pd-noomp/
```

## Run On Phone

```bash
adb shell chmod +x /data/local/tmp/llama-pd-noomp/llama-completion
adb shell 'cd /data/local/tmp/llama-pd-noomp && LD_LIBRARY_PATH=/data/local/tmp/llama-pd-noomp ./llama-completion -m /data/local/tmp/llama-pd-noomp/Qwen3-1.7b-2bit.gptq2_32.gguf -ngl 0 -t 4 -n 16 -p "1+1="' 
```
