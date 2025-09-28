# Tiny SIMD Core Library

This directory contains the core SIMD (Single Instruction Multiple Data) vector library implementation.

## Overview

The `tiny_simd.hpp` header file provides a high-performance, cross-platform SIMD vector computation engine with support for multiple architectures including ARM NEON and x86 SSE/AVX.

## Data Types

### Core Vector Class

- **`simd_vector<T, N>`**: Template class for SIMD vectors
  - `T`: Element type (arithmetic types: float, double, int32_t, int16_t, int8_t)
  - `N`: Vector size (number of elements)
  - Features automatic SIMD optimization detection
  - Properly aligned memory layout

### Type Aliases

#### Floating Point Vectors
- **`vec2f`**: `simd_vector<float, 2>` - 2D float vector
- **`vec3f`**: `simd_vector<float, 3>` - 3D float vector
- **`vec4f`**: `simd_vector<float, 4>` - 4D float vector (SIMD optimized)
- **`vec8f`**: `simd_vector<float, 8>` - 8D float vector (AVX optimized)

#### Double Precision Vectors
- **`vec2d`**: `simd_vector<double, 2>` - 2D double vector
- **`vec4d`**: `simd_vector<double, 4>` - 4D double vector

#### Integer Vectors
- **`vec4i`**: `simd_vector<int32_t, 4>` - 4D int32 vector (SIMD optimized)
- **`vec8i`**: `simd_vector<int32_t, 8>` - 8D int32 vector (AVX optimized)

#### Short Integer Vectors
- **`vec8s`**: `simd_vector<int16_t, 8>` - 8D int16 vector
- **`vec16s`**: `simd_vector<int16_t, 16>` - 16D int16 vector

#### Byte Vectors
- **`vec16b`**: `simd_vector<int8_t, 16>` - 16D int8 vector
- **`vec32b`**: `simd_vector<int8_t, 32>` - 32D int8 vector

## Supported SIMD Instructions

### Platform Support

#### ARM NEON
- **Float32x4**: 4-element float vectors with native NEON optimization
- **Int32x4**: 4-element int32 vectors with native NEON optimization
- Specialized instructions: `vaddq_f32`, `vsubq_f32`, `vmulq_f32`, `vminq_f32`, `vmaxq_f32`, etc.

#### x86 SSE/AVX
- **SSE**: 128-bit vectors (4 floats, 2 doubles, 4 int32s)
- **AVX**: 256-bit vectors (8 floats, 4 doubles)
- **AVX2**: Enhanced integer operations

### Basic Operations

#### Arithmetic Operations
- **Addition**: `+`, `+=` (vector-vector, vector-scalar)
- **Subtraction**: `-`, `-=` (vector-vector, vector-scalar)
- **Multiplication**: `*`, `*=` (element-wise, vector-vector, vector-scalar)
- **Division**: `/`, `/=` (element-wise, vector-vector, vector-scalar)
- **Negation**: unary `-`

#### Comparison Operations
- **Equality**: `==`, `!=`
- Element-wise comparison with SIMD acceleration

#### Memory Operations
- **Load**: `simd_vector(ptr)`, `load_aligned(ptr)`
- **Store**: `store(ptr)`, `store_aligned(ptr)`
- **Access**: `operator[]`, `data()`

### Vector Mathematical Functions

#### Basic Vector Operations
- **`dot(a, b)`**: Dot product of two vectors
- **`length(v)`**: Vector magnitude/length
- **`length_squared(v)`**: Squared vector length (avoids sqrt)
- **`normalize(v)`**: Unit vector in same direction
- **`distance(a, b)`**: Distance between two points
- **`distance_squared(a, b)`**: Squared distance (avoids sqrt)

#### Advanced Vector Operations
- **`lerp(a, b, t)`**: Linear interpolation between vectors
- **`project(a, b)`**: Vector projection (a projected onto b)
- **`reflect(v, n)`**: Vector reflection across normal
- **`cross(a, b)`**: Cross product (3D vectors only)

#### Element-wise Functions
- **`min(a, b)`**: Element-wise minimum
- **`max(a, b)`**: Element-wise maximum
- **`clamp(v, min, max)`**: Clamp values to range
- **`abs(v)`**: Element-wise absolute value

## SIMD Optimization Features

### Automatic Detection
- Runtime detection of available SIMD instruction sets
- Compile-time optimization flags
- Fallback to scalar implementation when SIMD unavailable

### ARM NEON Specializations
- **vec4f**: Full NEON float32x4_t optimization
- **vec4i**: Full NEON int32x4_t optimization
- Optimized dot product, min/max, abs operations
- Efficient reciprocal division approximation

### Memory Alignment
- Automatic alignment to SIMD requirements (16-byte for SSE/NEON, 32-byte for AVX)
- Aligned load/store operations for maximum performance

## Configuration

### Compile-Time Flags
- **`TINY_SIMD_ARM_NEON`**: Enable ARM NEON optimizations
- **`TINY_SIMD_X86_SSE`**: Enable x86 SSE optimizations
- **`TINY_SIMD_X86_AVX`**: Enable x86 AVX optimizations
- **`TINY_SIMD_X86_AVX2`**: Enable x86 AVX2 optimizations

### Runtime Configuration
- `config::has_neon`: Runtime NEON availability check
- `config::has_sse/avx/avx2`: Runtime x86 SIMD availability
- `config::max_vector_size_*`: Maximum supported vector sizes per type
- `config::simd_alignment`: Required memory alignment

## Usage Example

```cpp
#include "core/tiny_simd.hpp"
using namespace tiny_simd;

// Create vectors
vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
vec4f b{5.0f, 6.0f, 7.0f, 8.0f};

// Basic operations
vec4f sum = a + b;           // SIMD optimized addition
vec4f scaled = a * 2.0f;     // Scalar multiplication
float dp = dot(a, b);        // SIMD optimized dot product

// Vector math
float len = length(a);       // Vector magnitude
vec4f normalized = normalize(a); // Unit vector
vec4f clamped = clamp(a, 0.0f, 10.0f); // Clamp to range
```

## Performance Notes

- SIMD-optimized operations provide 2-8x performance improvement over scalar code
- ARM NEON optimizations available for float and int32 4-element vectors
- Memory alignment is crucial for maximum SIMD performance
- Use `load_aligned()` and `store_aligned()` when possible for best performance