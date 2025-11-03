#!/bin/bash
# Script to build and test Tiny SIMD Engine
#
# Usage:
#   ./scripts/build_and_test.sh [options]
#
# Options:
#   -c, --clean       Clean build (removes build directory)
#   -r, --release     Build in Release mode (default: Debug)
#   -o, --opencv      Build with OpenCV support
#   -t, --test-only   Skip build, only run tests
#   -f, --filter      GTest filter pattern (e.g., "ConversionTest.*")
#   -v, --verbose     Verbose test output
#   -h, --help        Show this help message
#
# Examples:
#   ./scripts/build_and_test.sh                    # Debug build + run all tests
#   ./scripts/build_and_test.sh -c -r              # Clean Release build + tests
#   ./scripts/build_and_test.sh -f "NEON*"         # Build + run only NEON tests
#   ./scripts/build_and_test.sh -t                 # Skip build, run tests only
#   ./scripts/build_and_test.sh -o                 # Build with OpenCV + tests

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
CLEAN_BUILD=false
BUILD_TYPE="Debug"
WITH_OPENCV=false
TEST_ONLY=false
GTEST_FILTER=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -o|--opencv)
            WITH_OPENCV=true
            shift
            ;;
        -t|--test-only)
            TEST_ONLY=true
            shift
            ;;
        -f|--filter)
            GTEST_FILTER="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print colored messages
print_header() {
    echo -e "${BLUE}===================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Detect number of CPU cores
if command -v nproc &> /dev/null; then
    NCORES=$(nproc)
elif command -v sysctl &> /dev/null; then
    NCORES=$(sysctl -n hw.ncpu)
else
    NCORES=4
fi

# Start
print_header "Tiny SIMD Engine - Build and Test"
echo ""

# Show configuration
print_info "Configuration:"
echo "  Build Type:    $BUILD_TYPE"
echo "  Clean Build:   $CLEAN_BUILD"
echo "  OpenCV:        $WITH_OPENCV"
echo "  Test Only:     $TEST_ONLY"
echo "  CPU Cores:     $NCORES"
if [ -n "$GTEST_FILTER" ]; then
    echo "  Test Filter:   $GTEST_FILTER"
fi
echo ""

# Build phase
if [ "$TEST_ONLY" = false ]; then
    print_header "Build Phase"
    echo ""

    # Clean build if requested
    if [ "$CLEAN_BUILD" = true ]; then
        if [ -d "build" ]; then
            print_info "Cleaning build directory..."
            rm -rf build
            print_success "Build directory cleaned"
        else
            print_info "No build directory to clean"
        fi
        echo ""
    fi

    # Create build directory
    mkdir -p build
    cd build

    # Configure CMake
    print_info "Configuring CMake..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    if [ "$WITH_OPENCV" = true ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DTINY_SIMD_WITH_OPENCV=ON"

        # Check if OpenCV is available
        if command -v pkg-config &> /dev/null; then
            if pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null; then
                print_success "OpenCV detected"
            else
                print_warning "OpenCV not found via pkg-config"
                print_info "Install with: brew install opencv (macOS)"
            fi
        fi
    fi

    if cmake $CMAKE_ARGS .. > /dev/null 2>&1; then
        print_success "CMake configuration complete"
    else
        print_error "CMake configuration failed"
        cmake $CMAKE_ARGS ..  # Run again without suppressing output
        exit 1
    fi
    echo ""

    # Build
    print_info "Building with $NCORES cores..."
    BUILD_START=$(date +%s)

    if make -j$NCORES 2>&1 | tee build.log | grep -E "(Building|Linking|\[.*%\])"; then
        BUILD_END=$(date +%s)
        BUILD_TIME=$((BUILD_END - BUILD_START))
        echo ""
        print_success "Build completed in ${BUILD_TIME}s"
    else
        echo ""
        print_error "Build failed! Check build.log for details"
        exit 1
    fi

    cd ..
    echo ""
else
    print_info "Skipping build phase (--test-only)"
    echo ""
fi

# Test phase
print_header "Test Phase"
echo ""

# Check if test executable exists
if [ ! -f "build/bin/test/tiny_simd_unit_tests" ]; then
    print_error "Test executable not found: build/bin/test/tiny_simd_unit_tests"
    print_info "Run without --test-only to build first"
    exit 1
fi

# Run tests
cd build

print_info "Running unit tests..."
echo ""

# Build test command
TEST_CMD="./bin/test/tiny_simd_unit_tests"
if [ -n "$GTEST_FILTER" ]; then
    TEST_CMD="$TEST_CMD --gtest_filter=\"$GTEST_FILTER\""
    print_info "Filter: $GTEST_FILTER"
fi

if [ "$VERBOSE" = true ]; then
    TEST_CMD="$TEST_CMD --gtest_print_time=1"
fi

# Run tests and capture output
TEST_START=$(date +%s)
if eval $TEST_CMD 2>&1 | tee test.log; then
    TEST_END=$(date +%s)
    TEST_TIME=$((TEST_END - TEST_START))
    echo ""

    # Parse test results (compatible with both GNU and BSD grep)
    TOTAL_TESTS=$(grep -o '\[  PASSED  \] [0-9]* test' test.log | tail -1 | grep -o '[0-9]*' | tail -1 || echo "0")
    TEST_SUITES=$(grep -o '[0-9]* test suite' test.log | tail -1 | grep -o '[0-9]*' || echo "0")

    print_success "All tests passed! ($TOTAL_TESTS tests from $TEST_SUITES test suites in ${TEST_TIME}s)"

    # Show summary
    echo ""
    print_info "Test Summary:"
    grep -A 1 "Global test environment tear-down" test.log | tail -1 || true

else
    TEST_END=$(date +%s)
    TEST_TIME=$((TEST_END - TEST_START))
    echo ""
    print_error "Tests failed after ${TEST_TIME}s"

    # Show failed tests
    if grep -q "FAILED" test.log; then
        echo ""
        print_info "Failed tests:"
        grep -A 5 "FAILED" test.log || true
    fi

    exit 1
fi

cd ..
echo ""

# Final summary
print_header "Summary"
echo ""

if [ "$TEST_ONLY" = false ]; then
    echo "  Build Time:    ${BUILD_TIME}s"
fi
echo "  Test Time:     ${TEST_TIME}s"
echo "  Tests Passed:  $TOTAL_TESTS"
echo "  Test Suites:   $TEST_SUITES"
echo ""

print_success "Build and test completed successfully!"
echo ""

# Show useful commands
print_info "Useful commands:"
echo "  View build log:    less build/build.log"
echo "  View test log:     less build/test.log"
echo "  Run tests again:   cd build && ./bin/test/tiny_simd_unit_tests"
if [ "$WITH_OPENCV" = false ]; then
    echo "  Build with OpenCV: ./scripts/build_and_test.sh -o"
fi
echo ""
