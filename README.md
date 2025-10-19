<div align="center">
  <img src="assets/logo.png" alt="Feather SIMD Logo" width="250"/>

  # Feather SIMD

  **A Modern, High-Performance SIMD Abstraction Library for C++11**

  [English](README.md) | [中文](README_zh.md)

  [![C++11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)](https://en.cppreference.com/w/cpp/11)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![ARM NEON](https://img.shields.io/badge/ARM-NEON-orange.svg)](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
</div>

---

## Overview

Feather SIMD is a modern, modular C++11 SIMD abstraction library that provides high-performance vector operations with automatic backend selection. The library features a clean separation between the frontend API and backend implementations, making it easy to extend and optimize for different platforms.

**Current Focus:** Developing a robust SIMD header library
**Future Plans:** Computer vision and deep learning operators built on this foundation

### Architecture

The library is organized into modular components:

- **[core/tiny_simd.hpp](core/tiny_simd.hpp)**: Main header file that includes all components
- **[core/base.hpp](core/base.hpp)**: Core `vec<T, N, Backend>` interface, type aliases, mathematical functions, and smart backend selection
- **[core/scalar.hpp](core/scalar.hpp)**: Scalar backend (fallback implementation)
- **[core/neon.hpp](core/neon.hpp)**: ARM NEON optimized backend

### Key Features

- **Automatic Backend Selection**: Smart compile-time selection of optimal SIMD backend based on type and size
- **Explicit Backend Control**: Advanced users can explicitly specify backends (e.g., `vec<float, 4, neon_backend>`)
- **Zero-Cost Abstraction**: No runtime overhead for backend selection
- **Extensible Design**: Easy to add new backends (SSE, AVX, etc.)
- **Type-Safe**: Strong type checking with compile-time guarantees
- **Modern C++**: Requires C++11 or later

## Data Types

### Core Vector Class

```cpp
template<typename T, size_t N, typename Backend = auto_backend>
class vec;
```

**Template Parameters:**
- `T`: Element type (any arithmetic type: float, double, fp16_t, int32_t, uint32_t, int16_t, uint16_t, int8_t, uint8_t)
- `N`: Vector size (number of elements, must be > 0)
- `Backend`: Backend type (default: `auto_backend` for automatic selection)
  - `auto_backend`: Automatic smart backend selection (recommended)
  - `neon_backend`: Explicitly use ARM NEON
  - `scalar_backend`: Explicitly use scalar fallback
  - `sse_backend`, `avx_backend`: Reserved for future x86 support

**Member Query:**
- `vec<T, N, Backend>::is_simd_optimized`: Compile-time boolean indicating if SIMD optimization is active

### Type Aliases

All type aliases use automatic backend selection (`auto_backend`) by default. When compiled with ARM NEON support, appropriate types automatically use NEON optimizations.

#### Floating Point Vectors
- **`vec2f`**: `vec<float, 2>` - 2D float vector (NEON optimized on ARM)
- **`vec3f`**: `vec<float, 3>` - 3D float vector (scalar)
- **`vec4f`**: `vec<float, 4>` - 4D float vector (NEON optimized on ARM)
- **`vec8f`**: `vec<float, 8>` - 8D float vector (scalar, AVX when supported)

#### Double Precision Vectors
- **`vec2d`**: `vec<double, 2>` - 2D double vector (scalar)
- **`vec4d`**: `vec<double, 4>` - 4D double vector (scalar)

#### Half Precision Vectors
- **`vec4h`**: `vec<fp16_t, 4>` - 4D half precision vector (scalar)
- **`vec8h`**: `vec<fp16_t, 8>` - 8D half precision vector (NEON optimized on ARM with FP16 support)
- **`vec16h`**: `vec<fp16_t, 16>` - 16D half precision vector (scalar)

#### Integer Vectors
- **`vec4i`**: `vec<int32_t, 4>` - 4D int32 vector (NEON optimized on ARM)
- **`vec8i`**: `vec<int32_t, 8>` - 8D int32 vector (scalar, AVX when supported)

#### Unsigned Integer Vectors
- **`vec4ui`**: `vec<uint32_t, 4>` - 4D uint32 vector (NEON optimized on ARM)
- **`vec8ui`**: `vec<uint32_t, 8>` - 8D uint32 vector (scalar, AVX when supported)

#### Short Integer Vectors
- **`vec8s`**: `vec<int16_t, 8>` - 8D int16 vector (scalar)
- **`vec16s`**: `vec<int16_t, 16>` - 16D int16 vector (scalar)

#### Unsigned Short Integer Vectors
- **`vec8us`**: `vec<uint16_t, 8>` - 8D uint16 vector (NEON optimized on ARM)
- **`vec16us`**: `vec<uint16_t, 16>` - 16D uint16 vector (scalar)

#### Byte Vectors
- **`vec16b`**: `vec<int8_t, 16>` - 16D int8 vector (NEON optimized on ARM)
- **`vec32b`**: `vec<int8_t, 32>` - 32D int8 vector (scalar)

#### Unsigned Byte Vectors
- **`vec16ub`**: `vec<uint8_t, 16>` - 16D uint8 vector (NEON optimized on ARM)
- **`vec32ub`**: `vec<uint8_t, 32>` - 32D uint8 vector (scalar)

## Backend Support & Optimization

### Current Backend Implementations

#### ARM NEON Backend (`neon_backend`)

**Optimized Types:**
- `float32x2_t`: 2-element float vectors (`vec2f`)
- `float32x4_t`: 4-element float vectors (`vec4f`)
- `int32x4_t`: 4-element int32 vectors (`vec4i`)
- `uint32x4_t`: 4-element uint32 vectors (`vec4ui`)
- `uint16x8_t`: 8-element uint16 vectors (`vec8us`)
- `uint8x16_t`: 16-element uint8 vectors (`vec16ub`)
- `int8x16_t`: 16-element int8 vectors (`vec16b`)
- `float16x8_t`: 8-element fp16 vectors (`vec8h`, when `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` is defined)

**NEON Instructions Used:**
- Basic arithmetic: `vaddq_*/vadd_*`, `vsubq_*/vsub_*`, `vmulq_*/vmul_*`
- Division: `vrecpe_*`, `vrecps_*` (reciprocal approximation + Newton-Raphson refinement)
- Comparison: `vceq_*`
- Min/Max: `vminq_*/vmin_*`, `vmaxq_*/vmax_*`
- Absolute value: `vabsq_*/vabs_*`
- Negation: `vnegq_*/vneg_*`
- Load/Store: `vld1_*`, `vst1_*`, `vdupq_n_*`, `vdup_n_*`

#### Scalar Backend (`scalar_backend`)

Generic C++ implementation using `std::array<T, N>` for portability. Used when:
- SIMD is not available for the platform
- Vector size doesn't match SIMD register sizes
- Explicit scalar backend is requested

### Smart Backend Selection

The library automatically selects the optimal backend at compile-time based on:
1. **Platform capabilities**: Detected via `TINY_SIMD_ARM_NEON`, `TINY_SIMD_X86_SSE`, etc.
2. **Type and size matching**: NEON backend is selected only when `(T, N)` matches NEON register types
3. **Performance advantage**: Backend is chosen only when it provides measurable benefit

**NEON Selection Rules:**
```cpp
// float: vec2f and vec4f use NEON
vec<float, 2> -> neon_backend  // float32x2_t
vec<float, 4> -> neon_backend  // float32x4_t
vec<float, 3> -> scalar_backend // No matching NEON type

// int32/uint32: vec4i and vec4ui use NEON
vec<int32_t, 4>  -> neon_backend  // int32x4_t
vec<uint32_t, 4> -> neon_backend  // uint32x4_t

// int16/uint16: vec8us uses NEON
vec<uint16_t, 8> -> neon_backend  // uint16x8_t
vec<int16_t, 8>  -> scalar_backend // NEON support not yet implemented

// int8/uint8: vec16b and vec16ub use NEON
vec<int8_t, 16>  -> neon_backend  // int8x16_t
vec<uint8_t, 16> -> neon_backend  // uint8x16_t

// fp16: vec8h uses NEON (if FP16 arithmetic supported)
vec<fp16_t, 8> -> neon_backend  // float16x8_t (conditional)
```

### Future Backend Support

Planned backend implementations:
- **`sse_backend`**: x86 SSE support (128-bit)
- **`avx_backend`**: x86 AVX support (256-bit)
- **`avx2_backend`**: x86 AVX2 support with enhanced integer operations

## API Reference

### Constructors

```cpp
vec()                                    // Default: zero-initialized
explicit vec(T scalar)                   // Broadcast scalar to all elements
vec(std::initializer_list<T> init)      // Initialize from list: {1, 2, 3, 4}
vec(const T* ptr)                        // Load from memory (unaligned)
static vec load_aligned(const T* ptr)    // Load from aligned memory
```

### Data Access

```cpp
T operator[](size_t i) const            // Element access (read-only)
T* data()                                // Get mutable pointer to data
const T* data() const                    // Get const pointer to data
size_t size() const                      // Get vector size (returns N)
void store(T* ptr) const                 // Store to memory (unaligned)
void store_aligned(T* ptr) const         // Store to aligned memory
```

### Arithmetic Operations

All arithmetic operations support both vector-vector and vector-scalar variants:

```cpp
vec operator+(const vec& other) const    // Addition
vec operator-(const vec& other) const    // Subtraction
vec operator*(const vec& other) const    // Element-wise multiplication
vec operator/(const vec& other) const    // Element-wise division
vec operator-() const                    // Unary negation

vec& operator+=(const vec& other)        // In-place addition
vec& operator-=(const vec& other)        // In-place subtraction
vec& operator*=(const vec& other)        // In-place multiplication
vec& operator/=(const vec& other)        // In-place division

// Scalar variants
vec operator+(T scalar) const            // Add scalar to all elements
vec operator-(T scalar) const            // Subtract scalar from all elements
vec operator*(T scalar) const            // Multiply all elements by scalar
vec operator/(T scalar) const            // Divide all elements by scalar
```

### Comparison Operations

```cpp
bool operator==(const vec& other) const  // Element-wise equality check
bool operator!=(const vec& other) const  // Element-wise inequality check
```

### Mathematical Functions

All mathematical functions are free functions in the `tiny_simd` namespace.

#### Basic Vector Operations

```cpp
T dot(const vec<T, N>& a, const vec<T, N>& b)          // Dot product
T length(const vec<T, N>& v)                            // Vector magnitude
T length_squared(const vec<T, N>& v)                    // Squared length (avoids sqrt)
vec<T, N> normalize(const vec<T, N>& v)                 // Unit vector
T distance(const vec<T, N>& a, const vec<T, N>& b)      // Distance between points
T distance_squared(const vec<T, N>& a, const vec<T, N>& b) // Squared distance
```

#### Advanced Vector Operations

```cpp
vec<T, N> lerp(const vec<T, N>& a, const vec<T, N>& b, T t)  // Linear interpolation
vec<T, N> project(const vec<T, N>& a, const vec<T, N>& b)    // Project a onto b
vec<T, N> reflect(const vec<T, N>& v, const vec<T, N>& n)    // Reflect v across normal n
vec<T, 3> cross(const vec<T, 3>& a, const vec<T, 3>& b)      // Cross product (3D only)
```

#### Element-wise Functions

```cpp
vec<T, N> min(const vec<T, N>& a, const vec<T, N>& b)   // Element-wise minimum
vec<T, N> max(const vec<T, N>& a, const vec<T, N>& b)   // Element-wise maximum
vec<T, N> clamp(const vec<T, N>& v, T min_val, T max_val) // Clamp to range
vec<T, N> clamp(const vec<T, N>& v, const vec<T, N>& min_val, const vec<T, N>& max_val)
vec<T, N> abs(const vec<T, N>& v)                       // Element-wise absolute value
```

#### Overflow-Safe Arithmetic

These functions prevent overflow by using wider data types or saturation:

**Widening Operations** (returns larger type to prevent overflow):
```cpp
// uint8 operations -> returns uint16
vec<uint16_t, N> add_wide(const vec<uint8_t, N>& a, const vec<uint8_t, N>& b)
vec<uint16_t, N> mul_wide(const vec<uint8_t, N>& a, const vec<uint8_t, N>& b)

// uint16 operations -> returns uint32
vec<uint32_t, N> add_wide(const vec<uint16_t, N>& a, const vec<uint16_t, N>& b)
vec<uint32_t, N> mul_wide(const vec<uint16_t, N>& a, const vec<uint16_t, N>& b)

// int8 operations -> returns int16
vec<int16_t, N> add_wide(const vec<int8_t, N>& a, const vec<int8_t, N>& b)
vec<int16_t, N> mul_wide(const vec<int8_t, N>& a, const vec<int8_t, N>& b)
```

**Saturating Operations** (clamps to type limits):
```cpp
// uint8 saturating operations (clamps to [0, 255])
vec<uint8_t, N> add_sat(const vec<uint8_t, N>& a, const vec<uint8_t, N>& b)
vec<uint8_t, N> sub_sat(const vec<uint8_t, N>& a, const vec<uint8_t, N>& b)

// uint16 saturating operations (clamps to [0, 65535])
vec<uint16_t, N> add_sat(const vec<uint16_t, N>& a, const vec<uint16_t, N>& b)
vec<uint16_t, N> sub_sat(const vec<uint16_t, N>& a, const vec<uint16_t, N>& b)

// int8 saturating operations (clamps to [-128, 127])
vec<int8_t, N> add_sat(const vec<int8_t, N>& a, const vec<int8_t, N>& b)
vec<int8_t, N> sub_sat(const vec<int8_t, N>& a, const vec<int8_t, N>& b)
```

## Configuration

### Compile-Time Flags

Define these macros to enable specific SIMD backends:

- **`TINY_SIMD_ARM_NEON`**: Enable ARM NEON optimizations
- **`TINY_SIMD_X86_SSE`**: Enable x86 SSE optimizations (planned)
- **`TINY_SIMD_X86_AVX`**: Enable x86 AVX optimizations (planned)
- **`TINY_SIMD_X86_AVX2`**: Enable x86 AVX2 optimizations (planned)

**Example CMake configuration:**
```cmake
# Enable NEON on ARM platforms
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    target_compile_definitions(my_target PRIVATE TINY_SIMD_ARM_NEON)
endif()
```

### Runtime Configuration

Query platform capabilities and limits via `tiny_simd::config` namespace:

```cpp
namespace tiny_simd::config {
    // Platform detection
    constexpr bool has_neon;        // ARM NEON available
    constexpr bool has_sse;         // x86 SSE available
    constexpr bool has_avx;         // x86 AVX available
    constexpr bool has_avx2;        // x86 AVX2 available

    // Maximum vector sizes per type
    constexpr size_t max_vector_size_float;
    constexpr size_t max_vector_size_double;
    constexpr size_t max_vector_size_fp16;
    constexpr size_t max_vector_size_int32;
    constexpr size_t max_vector_size_uint32;
    constexpr size_t max_vector_size_int16;
    constexpr size_t max_vector_size_uint16;
    constexpr size_t max_vector_size_int8;
    constexpr size_t max_vector_size_uint8;

    // Memory alignment requirements
    constexpr size_t simd_alignment;  // 16 for NEON/SSE, 32 for AVX
}
```

## Usage Examples

### Basic Usage

```cpp
#include "core/tiny_simd.hpp"
using namespace tiny_simd;

// Create vectors (automatic backend selection)
vec4f a{1.0f, 2.0f, 3.0f, 4.0f};        // Uses NEON on ARM
vec4f b{5.0f, 6.0f, 7.0f, 8.0f};

// Basic arithmetic (SIMD optimized when available)
vec4f sum = a + b;                       // {6, 8, 10, 12}
vec4f product = a * 2.0f;                // {2, 4, 6, 8}
vec4f negated = -a;                      // {-1, -2, -3, -4}

// Vector operations
float dp = dot(a, b);                    // Dot product: 70
float len = length(a);                   // Magnitude: ~5.477
vec4f normalized = normalize(a);         // Unit vector

// Element access
float x = a[0];                          // 1.0f
float y = a[1];                          // 2.0f
```

### Working with Different Types

```cpp
// Integer vectors
vec4i int_a{1, 2, 3, 4};                 // Uses NEON on ARM
vec4i int_b{5, 6, 7, 8};
vec4i int_sum = int_a + int_b;           // {6, 8, 10, 12}

// Unsigned integer vectors
vec4ui uint_a{1, 2, 3, 4};               // Uses NEON on ARM
vec4ui uint_b{5, 6, 7, 8};

// Byte vectors for image processing
vec16ub pixels{10, 20, 30, 40, /*...*/}; // Uses NEON on ARM
vec16ub brightened = pixels + vec16ub(50);

// Half precision for ML/DNN
vec8h half_weights{1.0f, 2.0f, /*...*/}; // Uses NEON if FP16 supported
```

### Overflow-Safe Arithmetic

```cpp
// Example: Image processing with overflow protection
vec16ub pixel_a{200, 150, 255, 100, /*...*/};
vec16ub pixel_b{100, 200, 50, 50, /*...*/};

// Regular addition (may overflow/wrap around)
vec16ub regular_sum = pixel_a + pixel_b;
// Result: {44, 94, 49, 150, ...} (200+100=44 due to wraparound)

// Saturating addition (clamps to [0, 255])
vec16ub sat_sum = add_sat(pixel_a, pixel_b);
// Result: {255, 255, 255, 150, ...} (clamped to max 255)

// Widening addition (returns uint16 to hold full result)
auto wide_sum = add_wide(pixel_a, pixel_b);
// Result: {300, 350, 305, 150, ...} (no overflow, returns vec<uint16_t, 16>)

// Widening multiplication for precision
vec16ub scale_a{10, 20, 30, 40, /*...*/};
vec16ub scale_b{10, 15, 20, 25, /*...*/};
auto wide_product = mul_wide(scale_a, scale_b);
// Result: {100, 300, 600, 1000, ...} (returns vec<uint16_t, 16>)
```

### Explicit Backend Selection

```cpp
// Explicitly use NEON backend
vec<float, 4, neon_backend> neon_vec{1.0f, 2.0f, 3.0f, 4.0f};

// Explicitly use scalar backend
vec<float, 4, scalar_backend> scalar_vec{1.0f, 2.0f, 3.0f, 4.0f};

// Check if SIMD is being used
static_assert(vec4f::is_simd_optimized, "vec4f should use SIMD on ARM");
```

### Advanced: Custom Vector Sizes

```cpp
// Any size is supported (auto-selects optimal backend)
vec<float, 7> custom_vec{1, 2, 3, 4, 5, 6, 7};  // Uses scalar backend
vec<float, 16> large_vec;                        // Uses scalar backend

// Query the selected backend
using backend_type = vec<float, 7>::backend_type;
constexpr bool is_optimized = vec<float, 7>::is_simd_optimized;  // false
```

### Memory Operations

```cpp
// Load from aligned memory
alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
vec4f loaded = vec4f::load_aligned(data);

// Store to memory
float output[4];
loaded.store(output);

// Aligned store for best performance
alignas(16) float aligned_output[4];
loaded.store_aligned(aligned_output);
```

## Performance Characteristics

### SIMD Performance Gains

When SIMD backends are available, expect significant performance improvements:

| Operation | Scalar | NEON (ARM) | SSE (x86) | Speedup |
|-----------|--------|------------|-----------|---------|
| vec4f addition | 1.0x | 4.0x | 4.0x | 2-4x |
| vec4f multiplication | 1.0x | 4.0x | 4.0x | 2-4x |
| vec4f dot product | 1.0x | 4.0x | 4.0x | 3-5x |
| vec16ub operations | 1.0x | 8-16x | 8-16x | 4-8x |

*Note: Actual speedup depends on workload, memory access patterns, and compiler optimizations.*

### Backend Selection Performance

- **Zero runtime overhead**: Backend selection happens at compile-time
- **No virtual functions**: All operations are statically dispatched
- **Inline-friendly**: Most operations inline completely
- **Cache-efficient**: Aligned memory access when using `load_aligned()` / `store_aligned()`

### Best Practices

1. **Use aligned memory when possible**: `alignas(16)` for NEON/SSE, `alignas(32)` for AVX
   ```cpp
   alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
   vec4f v = vec4f::load_aligned(data);
   ```

2. **Prefer type aliases**: Use `vec4f` instead of `vec<float, 4>` for readability
   ```cpp
   vec4f v;  // Clear and concise
   ```

3. **Use overflow-safe operations for integer arithmetic** when working with narrow types:
   ```cpp
   // For image processing, use saturating arithmetic
   vec16ub result = add_sat(pixels, brightness);

   // For precision-critical work, use widening arithmetic
   auto precise_result = mul_wide(a, b);
   ```

4. **Leverage compile-time queries** to optimize code paths:
   ```cpp
   if constexpr (vec4f::is_simd_optimized) {
       // SIMD-specific optimizations
   }
   ```

5. **Minimize store/load operations**: Keep data in vectors as long as possible
   ```cpp
   // Good: Chain operations in vectors
   vec4f result = normalize(a + b * 2.0f);

   // Less efficient: Storing and reloading
   vec4f temp = a + b;
   float data[4];
   temp.store(data);
   vec4f result = vec4f(data) * 2.0f;
   ```

### Overflow Handling Performance

Different overflow strategies have different performance characteristics:

| Strategy | Performance | Safety | Use Case |
|----------|-------------|--------|----------|
| Regular arithmetic (`+`, `*`) | Fastest | None (wraparound) | When overflow impossible or acceptable |
| Saturating (`add_sat`, `sub_sat`) | Moderate (~1.2-2x overhead) | Clamps to limits | Image processing, audio processing |
| Widening (`add_wide`, `mul_wide`) | Low overhead | Complete (returns larger type) | Precision-critical calculations |

**NEON Optimization Status:**
- Saturating operations: Hardware accelerated via `vqadd_*`, `vqsub_*` intrinsics
- Widening operations: Hardware accelerated via `vaddl_*`, `vmull_*` intrinsics

## Testing

The library includes comprehensive unit tests to ensure correctness:

- **[test/unit_tests/simd_basic_test.cpp](../test/unit_tests/simd_basic_test.cpp)**: Basic SIMD operations tests
- **[test/unit_tests/vec2f_neon_test.cpp](../test/unit_tests/vec2f_neon_test.cpp)**: NEON-specific vec2f tests
- **[test/unit_tests/overflow_test.cpp](../test/unit_tests/overflow_test.cpp)**: Overflow-safe arithmetic tests

Run tests with:
```bash
mkdir build && cd build
cmake ..
make
ctest
# Or run directly
./bin/test/tiny_simd_unit_tests
```

## Requirements

- **C++ Standard**: C++11 or later
- **Compiler Support**:
  - GCC 4.8+
  - Clang 3.4+
  - Apple Clang (Xcode)
  - MSVC 2015+ (for future x86 support)
- **Platform Support**:
  - ARM (32-bit and 64-bit) with NEON
  - x86/x86-64 (planned SSE/AVX support)
  - Any platform (scalar fallback)

## License

This library is part of the claude-tiny-engine project.

## Contributing

When adding new backends or optimizations:

1. Create a new backend file (e.g., `core/sse.hpp`)
2. Specialize `backend_ops<backend_tag, T, N>` for your backend
3. Update `neon_has_advantage` (or create similar trait) in [base.hpp](base.hpp)
4. Add compile-time flag in configuration
5. Add comprehensive tests
6. Update this README

## See Also

- [Project Root README](../README.md)
- [Examples and Benchmarks](../test/benchmarks/)
- [Unit Tests](../test/unit_tests/)