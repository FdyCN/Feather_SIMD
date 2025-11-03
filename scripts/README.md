# Build and Test Scripts

This directory contains utility scripts for building and testing the Tiny SIMD Engine.

## build_and_test.sh

Comprehensive script for building and testing the project with various options.

### Quick Start

```bash
# Basic usage - build and test
./scripts/build_and_test.sh

# Clean Release build + run all tests
./scripts/build_and_test.sh -c -r

# Run only specific tests without rebuilding
./scripts/build_and_test.sh -t -f "ConversionTest.*"
```

### Options

| Option | Long Form | Description |
|--------|-----------|-------------|
| `-c` | `--clean` | Clean build (removes build directory) |
| `-r` | `--release` | Build in Release mode (default: Debug) |
| `-o` | `--opencv` | Build with OpenCV support |
| `-t` | `--test-only` | Skip build, only run tests |
| `-f <pattern>` | `--filter <pattern>` | GTest filter pattern |
| `-v` | `--verbose` | Verbose test output |
| `-h` | `--help` | Show help message |

### Examples

#### Development Workflow

```bash
# Initial build and test
./scripts/build_and_test.sh

# Quick test after code changes (no rebuild)
./scripts/build_and_test.sh -t

# Run specific test suite
./scripts/build_and_test.sh -t -f "NEONDataTypesTest.*"

# Run all conversion tests
./scripts/build_and_test.sh -t -f "*Conversion*"
```

#### Release Testing

```bash
# Clean Release build with all tests
./scripts/build_and_test.sh -c -r

# Release build with OpenCV
./scripts/build_and_test.sh -c -r -o
```

#### Continuous Integration

```bash
# CI-friendly: Clean build, run all tests, verbose output
./scripts/build_and_test.sh -c -v

# Run only critical tests
./scripts/build_and_test.sh -c -f "NEON*:Conversion*:Width*"
```

### Test Filters

The `-f` option accepts [Google Test filter patterns](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests):

```bash
# Run all tests in a test suite
./scripts/build_and_test.sh -t -f "ConversionTest.*"

# Run specific test
./scripts/build_and_test.sh -t -f "ConversionTest.FP16ToFP32_Basic"

# Run multiple test suites
./scripts/build_and_test.sh -t -f "ConversionTest.*:WidthConversionTest.*"

# Run tests matching wildcard
./scripts/build_and_test.sh -t -f "*NEON*"

# Exclude tests (negative filter)
./scripts/build_and_test.sh -t -f "*-*Slow*"
```

### Output

The script provides:
- **Colored output** for easy reading
- **Build logs** saved to `build/build.log`
- **Test logs** saved to `build/test.log`
- **Summary** with timing and test counts

Example output:
```
===================================================================
Tiny SIMD Engine - Build and Test
===================================================================

ℹ Configuration:
  Build Type:    Debug
  Clean Build:   false
  OpenCV:        false
  Test Only:     false
  CPU Cores:     16

===================================================================
Build Phase
===================================================================

ℹ Configuring CMake...
✓ CMake configuration complete

ℹ Building with 16 cores...
✓ Build completed in 2s

===================================================================
Test Phase
===================================================================

ℹ Running unit tests...
✓ All tests passed! (98 tests from 15 test suites in 0s)

===================================================================
Summary
===================================================================

  Build Time:    2s
  Test Time:     0s
  Tests Passed:  98
  Test Suites:   15

✓ Build and test completed successfully!
```

## build_with_opencv.sh

Legacy script specifically for building with OpenCV support. For new workflows, prefer `build_and_test.sh -o`.

## check_dependencies.sh

Script to check system dependencies and verify the development environment.

---

## Tips

### Faster Development Cycle

```bash
# 1. Full build once
./scripts/build_and_test.sh

# 2. Make code changes

# 3. Quick rebuild and test (only changed files)
./scripts/build_and_test.sh

# 4. Or just run tests if only test code changed
./scripts/build_and_test.sh -t
```

### Debugging Failed Tests

```bash
# Run failing test with verbose output
./scripts/build_and_test.sh -t -v -f "FailingTest.*"

# Check test log for details
less build/test.log

# Run test directly for interactive debugging
cd build
./bin/test/tiny_simd_unit_tests --gtest_filter="FailingTest.*"
```

### Performance Testing

```bash
# Release build for accurate performance measurement
./scripts/build_and_test.sh -c -r

# Run benchmarks if available
cd build
./bin/benchmarks/simd_benchmarks
```

## Script Maintenance

The script is designed to be:
- **Cross-platform**: Works on macOS and Linux
- **Portable**: Uses standard bash features
- **Robust**: Error handling with `set -e`
- **Informative**: Clear output with colors and progress

For issues or improvements, check the script comments or contact the maintainers.
