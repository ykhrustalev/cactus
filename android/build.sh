#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANDROID_DIR="$PROJECT_ROOT/android"

UPLOAD=false
for arg in "$@"; do
    case $arg in
        --upload)
            UPLOAD=true
            ;;
    esac
done

ANDROID_PLATFORM=${ANDROID_PLATFORM:-android-21}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
BUILD_DIR="$ANDROID_DIR/build"

if [ -z "$ANDROID_NDK_HOME" ]; then
    if [ -n "$ANDROID_HOME" ]; then
        ANDROID_NDK_HOME=$(ls -d "$ANDROID_HOME/ndk/"* 2>/dev/null | sort -V | tail -1)
    elif [ -d "$HOME/Library/Android/sdk" ]; then
        ANDROID_NDK_HOME=$(ls -d "$HOME/Library/Android/sdk/ndk/"* 2>/dev/null | sort -V | tail -1)
    fi
fi

if [ -z "$ANDROID_NDK_HOME" ] || [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "Error: Android NDK not found."
    echo "Set ANDROID_NDK_HOME or install NDK via Android SDK Manager"
    exit 1
fi

echo "Using NDK: $ANDROID_NDK_HOME"
CMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"

if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found, please install it"
    exit 1
fi

n_cpu=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

ABI="arm64-v8a"

echo "Building Cactus for Android ($ABI)..."
echo "Build type: $CMAKE_BUILD_TYPE"
echo "Using $n_cpu CPU cores"
echo "Android CMakeLists.txt: $ANDROID_DIR/CMakeLists.txt"

cmake -DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE" \
      -DANDROID_ABI="$ABI" \
      -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
      -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
      -S "$ANDROID_DIR" \
      -B "$BUILD_DIR"

cmake --build "$BUILD_DIR" --config "$CMAKE_BUILD_TYPE" -j "$n_cpu"

cp "$BUILD_DIR/lib/libcactus.so" "$ANDROID_DIR/" 2>/dev/null || \
   cp "$BUILD_DIR/libcactus.so" "$ANDROID_DIR" 2>/dev/null || \
   { echo "Error: Could not find libcactus.so"; exit 1; }

cp "$BUILD_DIR/lib/libcactus_static.a" "$ANDROID_DIR/libcactus.a" 2>/dev/null || \
   cp "$BUILD_DIR/libcactus_static.a" "$ANDROID_DIR/libcactus.a" 2>/dev/null || \
   { echo "Warning: Could not find libcactus_static.a"; }

cp "$BUILD_DIR/bin/cactus-bench" "$ANDROID_DIR/cactus-bench" 2>/dev/null || \
   cp "$BUILD_DIR/cactus-bench" "$ANDROID_DIR/cactus-bench" 2>/dev/null || \
   { echo "Warning: Could not find cactus-bench executable"; }

echo "Build complete!"
echo "Shared library location: $ANDROID_DIR/libcactus.so"
echo "Static library location: $ANDROID_DIR/libcactus.a"
echo "Benchmark executable location: $ANDROID_DIR/cactus-bench"

# Package and upload cactus-bench to S3
if [ "$UPLOAD" = true ]; then
    if [ -f "$ANDROID_DIR/cactus-bench" ]; then
        GIT_HASH=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)
        ZIP_NAME="cactus-bench-android-${GIT_HASH}.zip"
        ZIP_PATH="$ANDROID_DIR/$ZIP_NAME"

        echo "Packaging cactus-bench..."
        (cd "$ANDROID_DIR" && zip -j "$ZIP_NAME" cactus-bench)

        echo "Uploading $ZIP_NAME to s3://liquid-eb-resources..."
        aws s3 cp "$ZIP_PATH" "s3://liquid-eb-resources/$ZIP_NAME" --profile dev

        echo "Upload complete: s3://liquid-eb-resources/$ZIP_NAME"
    else
        echo "Error: cactus-bench not found, cannot upload"
        exit 1
    fi
fi