#!/bin/bash
set -e

MODEL="lfm2-1.2b"

adb push --sync "weights/${MODEL}" "/data/local/tmp/weights/"
adb push android/cactus-bench /data/local/tmp/
adb shell chmod +x /data/local/tmp/cactus-bench

echo "=== Prefill Benchmarks ==="
for tokens in 128 256 512 1024 2048 4096; do
    echo "Running prefill ${tokens}..."
    adb shell "/data/local/tmp/cactus-bench /data/local/tmp/weights/${MODEL} --action prefill --tokens ${tokens}"
done

echo ""
echo "=== Decode Benchmarks ==="
for offset in 128 256 512 1024 2048 4096; do
    echo "Running decode offset=${offset} tokens=100..."
    adb shell "/data/local/tmp/cactus-bench /data/local/tmp/weights/${MODEL} --action decode --offset ${offset} --tokens 100"
done

echo ""
echo "Done!"
