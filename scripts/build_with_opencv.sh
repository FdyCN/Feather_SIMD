#!/bin/bash
# Script to build Tiny SIMD Engine with OpenCV support
#
# Usage:
#   ./scripts/build_with_opencv.sh
#
# Prerequisites:
#   - OpenCV installed (brew install opencv on macOS)

set -e

echo "=== Building Tiny SIMD Engine with OpenCV Support ==="
echo ""

# Check if OpenCV is installed
if command -v pkg-config &> /dev/null; then
    if pkg-config --exists opencv4 2>/dev/null; then
        echo "✓ OpenCV detected: $(pkg-config --modversion opencv4)"
    elif pkg-config --exists opencv 2>/dev/null; then
        echo "✓ OpenCV detected: $(pkg-config --modversion opencv)"
    else
        echo "⚠ Warning: OpenCV not found via pkg-config"
        echo "  Install with: brew install opencv (macOS)"
        echo "  Or: sudo apt-get install libopencv-dev (Ubuntu/Debian)"
        echo ""
        echo "Continuing anyway..."
    fi
else
    echo "⚠ pkg-config not found, cannot verify OpenCV installation"
fi

echo ""

# Create build directory
if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi

mkdir -p build
cd build

# Configure with OpenCV support
echo "Configuring CMake with OpenCV support..."
cmake -DTINY_SIMD_WITH_OPENCV=ON ..

echo ""
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=== Build Complete ==="
echo ""
echo "To run tests:"
echo "  cd build"
echo "  ./bin/test/tiny_simd_unit_tests"
echo ""
echo "To run only OpenCV comparison tests:"
echo "  ./bin/test/tiny_simd_unit_tests --gtest_filter=\"OpenCVComparison.*\""
echo ""
