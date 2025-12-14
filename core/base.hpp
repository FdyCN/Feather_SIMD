#ifndef TINY_SIMD_BASE_HPP
#define TINY_SIMD_BASE_HPP

#include <cstddef>
#include <type_traits>
#include <cstdint>
#include <array>
#include <initializer_list>
#include <cassert>
#include <cmath>
#include <algorithm>

// C++11 compatibility check
#if __cplusplus < 201103L
    #error "Tiny SIMD Engine requires C++11 or later"
#endif

// Half precision float type definition
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE)
    using fp16_t = __fp16;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    using fp16_t = uint16_t;
#else
    using fp16_t = uint16_t;
#endif

namespace tiny_simd {

//=============================================================================
// Type Traits for fp16_t Support
//=============================================================================

// Helper: Check if type is valid for SIMD operations
// std::is_arithmetic doesn't recognize __fp16, so we need custom trait
template<typename T>
struct is_simd_arithmetic {
    static constexpr bool value = std::is_arithmetic<T>::value;
};

// Specialize for fp16_t when it's __fp16
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE)
template<>
struct is_simd_arithmetic<fp16_t> {
    static constexpr bool value = true;
};
#endif

//=============================================================================
// Backend Tags - Backend type markers
//=============================================================================

struct scalar_backend {};  // Scalar backend (fallback)
struct neon_backend {};    // ARM NEON backend
struct sse_backend {};     // x86 SSE backend (future)
struct avx_backend {};     // x86 AVX backend (future)
struct avx2_backend {};    // x86 AVX2 backend
struct auto_backend {};    // Automatic backend selection

//=============================================================================
// Platform Detection Configuration
//=============================================================================

namespace config {

// Platform detection macros
#ifdef TINY_SIMD_ARM_NEON
static constexpr bool has_neon = true;
#else
static constexpr bool has_neon = false;
#endif

#ifdef TINY_SIMD_X86_SSE
static constexpr bool has_sse = true;
#else
static constexpr bool has_sse = false;
#endif

#ifdef TINY_SIMD_X86_AVX
static constexpr bool has_avx = true;
#else
static constexpr bool has_avx = false;
#endif

#ifdef TINY_SIMD_X86_AVX2
static constexpr bool has_avx2 = true;
#else
static constexpr bool has_avx2 = false;
#endif

// Vector size configuration (based on platform)
#ifdef TINY_SIMD_ARM_NEON
static constexpr size_t max_vector_size_float = 4;
static constexpr size_t max_vector_size_double = 2;
static constexpr size_t max_vector_size_fp16 = 8;
static constexpr size_t max_vector_size_int32 = 4;
static constexpr size_t max_vector_size_uint32 = 4;
static constexpr size_t max_vector_size_int16 = 8;
static constexpr size_t max_vector_size_uint16 = 8;
static constexpr size_t max_vector_size_int8 = 16;
static constexpr size_t max_vector_size_uint8 = 16;
#elif defined(TINY_SIMD_X86_AVX2)
static constexpr size_t max_vector_size_float = 8;
static constexpr size_t max_vector_size_double = 4;
static constexpr size_t max_vector_size_fp16 = 16;
static constexpr size_t max_vector_size_int32 = 8;
static constexpr size_t max_vector_size_uint32 = 8;
static constexpr size_t max_vector_size_int16 = 16;
static constexpr size_t max_vector_size_uint16 = 16;
static constexpr size_t max_vector_size_int8 = 32;
static constexpr size_t max_vector_size_uint8 = 32;
#elif defined(TINY_SIMD_X86_AVX)
static constexpr size_t max_vector_size_float = 8;
static constexpr size_t max_vector_size_double = 4;
static constexpr size_t max_vector_size_fp16 = 16;
static constexpr size_t max_vector_size_int32 = 4;
static constexpr size_t max_vector_size_uint32 = 4;
static constexpr size_t max_vector_size_int16 = 8;
static constexpr size_t max_vector_size_uint16 = 8;
static constexpr size_t max_vector_size_int8 = 16;
static constexpr size_t max_vector_size_uint8 = 16;
#elif defined(TINY_SIMD_X86_SSE)
static constexpr size_t max_vector_size_float = 4;
static constexpr size_t max_vector_size_double = 2;
static constexpr size_t max_vector_size_fp16 = 8;
static constexpr size_t max_vector_size_int32 = 4;
static constexpr size_t max_vector_size_uint32 = 4;
static constexpr size_t max_vector_size_int16 = 8;
static constexpr size_t max_vector_size_uint16 = 8;
static constexpr size_t max_vector_size_int8 = 16;
static constexpr size_t max_vector_size_uint8 = 16;
#else
static constexpr size_t max_vector_size_float = 1;
static constexpr size_t max_vector_size_double = 1;
static constexpr size_t max_vector_size_fp16 = 1;
static constexpr size_t max_vector_size_int32 = 1;
static constexpr size_t max_vector_size_uint32 = 1;
static constexpr size_t max_vector_size_int16 = 1;
static constexpr size_t max_vector_size_uint16 = 1;
static constexpr size_t max_vector_size_int8 = 1;
static constexpr size_t max_vector_size_uint8 = 1;
#endif

// Alignment requirements
#if defined(TINY_SIMD_ARM_NEON) || defined(TINY_SIMD_X86_SSE)
static constexpr size_t simd_alignment = 16;
#elif defined(TINY_SIMD_X86_AVX)
static constexpr size_t simd_alignment = 32;
#else
static constexpr size_t simd_alignment = sizeof(void*);
#endif

} // namespace config

//=============================================================================
// Smart Backend Selection Logic
//=============================================================================

// Helper: Check if NEON has advantage for given type and size
template<typename T, size_t N>
struct neon_has_advantage {
#ifdef TINY_SIMD_ARM_NEON
    static constexpr bool value =
        // float: N=2 or N=4 has NEON advantage
        (std::is_same<T, float>::value && (N == 2 || N == 4)) ||
        // int32/uint32: N=2 or N=4 has NEON advantage
        ((std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value) && (N == 2 || N == 4)) ||
        // int16/uint16: N=4 or N=8 has NEON advantage
        ((std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) && (N == 4 || N == 8)) ||
        // int8/uint8: N=8 or N=16 has NEON advantage
        ((std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) && (N == 8 || N == 16)) ||
        // fp16: N=4 or N=8 has NEON advantage (if supported)
        (std::is_same<T, fp16_t>::value && (N == 4 || N == 8));
#else
    static constexpr bool value = false;
#endif
};

// Helper: Check if AVX2 has advantage for given type and size
template<typename T, size_t N>
struct avx2_has_advantage {
#ifdef TINY_SIMD_X86_AVX2
    static constexpr bool value =
        // float: N=8
        (std::is_same<T, float>::value && N == 8) ||
        // double: N=4
        (std::is_same<T, double>::value && N == 4) ||
        // int32/uint32: N=8
        ((std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value) && N == 8) ||
        // int16/uint16: N=16
        ((std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) && N == 16) ||
        // int8/uint8: N=32
        ((std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) && N == 32);
#else
    static constexpr bool value = false;
#endif
};

// Default backend selection based on type and size
template<typename T, size_t N>
struct default_backend {
    using type = typename std::conditional<
        neon_has_advantage<T, N>::value,
        neon_backend,
        typename std::conditional<
            avx2_has_advantage<T, N>::value,
            avx2_backend,
            scalar_backend
        >::type
    >::type;
};

// Convenience alias
template<typename T, size_t N>
using default_backend_t = typename default_backend<T, N>::type;

//=============================================================================
// Backend Operations Interface (Pure Abstract)
//=============================================================================

// Forward declaration - to be specialized by each backend
template<typename Backend, typename T, size_t N>
struct backend_ops;

//=============================================================================
// Unified Vector Interface - vec<T, N, Backend>
//=============================================================================

template<typename T, size_t N, typename Backend = auto_backend>
class vec {
private:
    static_assert(N > 0, "Vector size must be positive");
    static_assert(is_simd_arithmetic<T>::value, "T must be arithmetic type or fp16_t");

    // Resolve actual backend: auto_backend -> smart selection, otherwise use explicit
    using actual_backend = typename std::conditional<
        std::is_same<Backend, auto_backend>::value,
        default_backend_t<T, N>,
        Backend
    >::type;

    // Backend operations interface
    using ops = backend_ops<actual_backend, T, N>;
    using reg_type = typename ops::reg_type;

    // Internal register storage
    reg_type reg_;

public:
    using value_type = T;
    using size_type = size_t;
    using backend_type = actual_backend;
    static constexpr size_type size_value = N;

    // Query whether SIMD optimization is used
    static constexpr bool is_simd_optimized = !std::is_same<actual_backend, scalar_backend>::value;

    //=============================================================================
    // Constructors - delegate to backend
    //=============================================================================

    vec() : reg_(ops::zero()) {}

    explicit vec(T scalar) : reg_(ops::set1(scalar)) {}

    vec(std::initializer_list<T> init) : reg_(ops::load_from_initializer(init)) {
        assert(init.size() == N);
    }

    vec(const T* ptr) : reg_(ops::load(ptr)) {}

    // Internal constructor from register
    explicit vec(reg_type reg) : reg_(reg) {}

    static vec load_aligned(const T* ptr) {
        return vec(ops::load_aligned(ptr));
    }

    //=============================================================================
    // Data Access - delegate to backend
    //=============================================================================

    T operator[](size_t i) const {
        assert(i < N);
        return ops::extract(reg_, i);
    }

    T* data() {
        // For mutable access, we need a temporary buffer
        static thread_local T buffer[N];
        ops::store(buffer, reg_);
        return buffer;
    }

    const T* data() const {
        static thread_local T buffer[N];
        ops::store(buffer, reg_);
        return buffer;
    }

    constexpr size_t size() const { return N; }

    void store(T* ptr) const {
        ops::store(ptr, reg_);
    }

    void store_aligned(T* ptr) const {
        ops::store_aligned(ptr, reg_);
    }

    //=============================================================================
    // Arithmetic Operations - delegate to backend
    //=============================================================================

    vec& operator+=(const vec& other) {
        reg_ = ops::add(reg_, other.reg_);
        return *this;
    }

    vec& operator-=(const vec& other) {
        reg_ = ops::sub(reg_, other.reg_);
        return *this;
    }

    vec& operator*=(const vec& other) {
        reg_ = ops::mul(reg_, other.reg_);
        return *this;
    }

    vec& operator/=(const vec& other) {
        reg_ = ops::div(reg_, other.reg_);
        return *this;
    }

    // Scalar operations
    vec& operator+=(T scalar) {
        return *this += vec(scalar);
    }

    vec& operator-=(T scalar) {
        return *this -= vec(scalar);
    }

    vec& operator*=(T scalar) {
        return *this *= vec(scalar);
    }

    vec& operator/=(T scalar) {
        return *this /= vec(scalar);
    }

    // Unary negation
    vec operator-() const {
        return vec(ops::neg(reg_));
    }

    //=============================================================================
    // Bitwise Operations (Integer only)
    //=============================================================================

    template<typename U = T>
    typename std::enable_if<std::is_integral<U>::value, vec&>::type
    operator&=(const vec& other) {
        reg_ = ops::bitwise_and(reg_, other.reg_);
        return *this;
    }

    template<typename U = T>
    typename std::enable_if<std::is_integral<U>::value, vec&>::type
    operator|=(const vec& other) {
        reg_ = ops::bitwise_or(reg_, other.reg_);
        return *this;
    }

    template<typename U = T>
    typename std::enable_if<std::is_integral<U>::value, vec&>::type
    operator^=(const vec& other) {
        reg_ = ops::bitwise_xor(reg_, other.reg_);
        return *this;
    }

    template<typename U = T>
    typename std::enable_if<std::is_integral<U>::value, vec>::type
    operator~() const {
        return vec(ops::bitwise_not(reg_));
    }

    //=============================================================================
    // Shift Operations (Integer only)
    //=============================================================================

    template<typename U = T>
    typename std::enable_if<std::is_integral<U>::value, vec&>::type
    operator<<=(int count) {
        reg_ = ops::shift_left(reg_, count);
        return *this;
    }

    template<typename U = T>
    typename std::enable_if<std::is_integral<U>::value, vec&>::type
    operator>>=(int count) {
        reg_ = ops::shift_right(reg_, count);
        return *this;
    }

    //=============================================================================
    // Comparison Operations
    //=============================================================================

    bool operator==(const vec& other) const {
        return ops::equal(reg_, other.reg_);
    }

    bool operator!=(const vec& other) const {
        return !(*this == other);
    }

    //=============================================================================
    // Access to internal register (for advanced users)
    //=============================================================================

    reg_type& reg() { return reg_; }
    const reg_type& reg() const { return reg_; }
};

template<typename T, size_t N, typename Backend>
constexpr typename vec<T, N, Backend>::size_type vec<T, N, Backend>::size_value;

template<typename T, size_t N, typename Backend>
constexpr bool vec<T, N, Backend>::is_simd_optimized;

//=============================================================================
// Binary Operators
//=============================================================================

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator+(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    vec<T, N, Backend> result = a;
    result += b;
    return result;
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator-(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    vec<T, N, Backend> result = a;
    result -= b;
    return result;
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator*(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    vec<T, N, Backend> result = a;
    result *= b;
    return result;
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator/(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    vec<T, N, Backend> result = a;
    result /= b;
    return result;
}

// Scalar operations
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator+(const vec<T, N, Backend>& v, T scalar) {
    return v + vec<T, N, Backend>(scalar);
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator+(T scalar, const vec<T, N, Backend>& v) {
    return vec<T, N, Backend>(scalar) + v;
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator-(const vec<T, N, Backend>& v, T scalar) {
    return v - vec<T, N, Backend>(scalar);
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator-(T scalar, const vec<T, N, Backend>& v) {
    return vec<T, N, Backend>(scalar) - v;
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator*(const vec<T, N, Backend>& v, T scalar) {
    return v * vec<T, N, Backend>(scalar);
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator*(T scalar, const vec<T, N, Backend>& v) {
    return vec<T, N, Backend>(scalar) * v;
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> operator/(const vec<T, N, Backend>& v, T scalar) {
    return v / vec<T, N, Backend>(scalar);
}

//=============================================================================
// Bitwise Operators (Integer only)
//=============================================================================

template<typename T, size_t N, typename Backend>
inline typename std::enable_if<std::is_integral<T>::value, vec<T, N, Backend>>::type
operator&(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    vec<T, N, Backend> result = a;
    result &= b;
    return result;
}

template<typename T, size_t N, typename Backend>
inline typename std::enable_if<std::is_integral<T>::value, vec<T, N, Backend>>::type
operator|(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    vec<T, N, Backend> result = a;
    result |= b;
    return result;
}

template<typename T, size_t N, typename Backend>
inline typename std::enable_if<std::is_integral<T>::value, vec<T, N, Backend>>::type
operator^(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    vec<T, N, Backend> result = a;
    result ^= b;
    return result;
}

template<typename T, size_t N, typename Backend>
inline typename std::enable_if<std::is_integral<T>::value, vec<T, N, Backend>>::type
bitwise_andnot(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    // Use backend optimization if available
    using ops = backend_ops<typename vec<T, N, Backend>::backend_type, T, N>;
    return vec<T, N, Backend>(ops::bitwise_andnot(a.reg(), b.reg()));
}

//=============================================================================
// Shift Operators (Integer only)
//=============================================================================

template<typename T, size_t N, typename Backend>
inline typename std::enable_if<std::is_integral<T>::value, vec<T, N, Backend>>::type
operator<<(const vec<T, N, Backend>& a, int count) {
    vec<T, N, Backend> result = a;
    result <<= count;
    return result;
}

template<typename T, size_t N, typename Backend>
inline typename std::enable_if<std::is_integral<T>::value, vec<T, N, Backend>>::type
operator>>(const vec<T, N, Backend>& a, int count) {
    vec<T, N, Backend> result = a;
    result >>= count;
    return result;
}

//=============================================================================
// Common Type Aliases (using auto backend)
//=============================================================================

using vec2f = vec<float, 2>;
using vec3f = vec<float, 3>;
using vec4f = vec<float, 4>;
using vec8f = vec<float, 8>;

using vec2d = vec<double, 2>;
using vec4d = vec<double, 4>;

using vec4h = vec<fp16_t, 4>;
using vec8h = vec<fp16_t, 8>;
using vec16h = vec<fp16_t, 16>;

using vec4i = vec<int32_t, 4>;
using vec8i = vec<int32_t, 8>;

using vec4ui = vec<uint32_t, 4>;
using vec8ui = vec<uint32_t, 8>;

using vec8s = vec<int16_t, 8>;
using vec16s = vec<int16_t, 16>;

using vec8us = vec<uint16_t, 8>;
using vec16us = vec<uint16_t, 16>;

using vec16b = vec<int8_t, 16>;
using vec32b = vec<int8_t, 32>;

using vec16ub = vec<uint8_t, 16>;
using vec32ub = vec<uint8_t, 32>;

//=============================================================================
// Mathematical Functions (Generic implementations, can be specialized)
//=============================================================================

// Dot product
template<typename T, size_t N, typename Backend>
inline T dot(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    T result = T{0};
    for (size_t i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Vector length squared
template<typename T, size_t N, typename Backend>
inline T length_squared(const vec<T, N, Backend>& v) {
    return dot(v, v);
}

// Vector length
template<typename T, size_t N, typename Backend>
inline T length(const vec<T, N, Backend>& v) {
    return std::sqrt(length_squared(v));
}

// Normalize
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> normalize(const vec<T, N, Backend>& v) {
    T len = length(v);
    if (len == T{0}) return v;
    return v / len;
}

// Distance
template<typename T, size_t N, typename Backend>
inline T distance(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    return length(a - b);
}

// Distance squared
template<typename T, size_t N, typename Backend>
inline T distance_squared(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    return length_squared(a - b);
}

// Linear interpolation
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> lerp(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b, T t) {
    return a + (b - a) * t;
}

// Project a onto b
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> project(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    T b_length_sq = length_squared(b);
    if (b_length_sq == T{0}) return vec<T, N, Backend>(T{0});
    return b * (dot(a, b) / b_length_sq);
}

// Reflect v across normal n
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> reflect(const vec<T, N, Backend>& v, const vec<T, N, Backend>& n) {
    return v - n * (T{2} * dot(v, n));
}

// Cross product (3D only)
template<typename T, typename Backend>
inline vec<T, 3, Backend> cross(const vec<T, 3, Backend>& a, const vec<T, 3, Backend>& b) {
    return vec<T, 3, Backend>({
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    });
}

// Min
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> min(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    // Use backend operations if available, otherwise generic implementation
    using ops = backend_ops<typename vec<T, N, Backend>::backend_type, T, N>;
    return vec<T, N, Backend>(ops::min(a.reg(), b.reg()));
}

// Max
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> max(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b) {
    // Use backend operations if available, otherwise generic implementation
    using ops = backend_ops<typename vec<T, N, Backend>::backend_type, T, N>;
    return vec<T, N, Backend>(ops::max(a.reg(), b.reg()));
}

// Clamp
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> clamp(const vec<T, N, Backend>& v, const vec<T, N, Backend>& min_val, const vec<T, N, Backend>& max_val) {
    return min(max(v, min_val), max_val);
}

template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> clamp(const vec<T, N, Backend>& v, T min_val, T max_val) {
    return clamp(v, vec<T, N, Backend>(min_val), vec<T, N, Backend>(max_val));
}

// Absolute value
template<typename T, size_t N, typename Backend>
inline vec<T, N, Backend> abs(const vec<T, N, Backend>& v) {
    using ops = backend_ops<typename vec<T, N, Backend>::backend_type, T, N>;
    return vec<T, N, Backend>(ops::abs(v.reg()));
}

// Fused Multiply-Add (FMA): a * b + c
// Only for floating point types
template<typename T, size_t N, typename Backend>
inline typename std::enable_if<std::is_floating_point<T>::value, vec<T, N, Backend>>::type
fma(const vec<T, N, Backend>& a, const vec<T, N, Backend>& b, const vec<T, N, Backend>& c) {
    using ops = backend_ops<typename vec<T, N, Backend>::backend_type, T, N>;
    return vec<T, N, Backend>(ops::fma(a.reg(), b.reg(), c.reg()));
}


//=============================================================================
// Vector Splitting Operations (向量拆分)
//=============================================================================
// Extract low/high halves of vectors - unified interface that calls backend_ops

// Unified interface for get_low - calls backend_ops
template<typename T, size_t N, typename Backend = auto_backend>
inline vec<T, N/2, Backend> get_low(const vec<T, N, Backend>& v) {
    static_assert(N % 2 == 0, "Vector size must be even for get_low");
    // Resolve auto_backend to actual backend type
    using actual_backend = typename vec<T, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, T, N>;
    auto low_reg = ops::get_low(v.reg());
    return vec<T, N/2, Backend>(low_reg);
}

// Unified interface for get_high - calls backend_ops
template<typename T, size_t N, typename Backend = auto_backend>
inline vec<T, N/2, Backend> get_high(const vec<T, N, Backend>& v) {
    static_assert(N % 2 == 0, "Vector size must be even for get_high");
    // Resolve auto_backend to actual backend type
    using actual_backend = typename vec<T, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, T, N>;
    auto high_reg = ops::get_high(v.reg());
    return vec<T, N/2, Backend>(high_reg);
}

//=============================================================================
// Type Conversion Operations (类型转换)
//=============================================================================
// Phase 1: Most commonly used conversions
//   1. fp16 <-> fp32 (half precision <-> single precision float)
//   2. int32 -> float32 (integer to float)
//   3. float32 -> int32 (float to integer with rounding)

//-----------------------------------------------------------------------------
// Float Precision Conversions: fp16 <-> fp32
//-----------------------------------------------------------------------------

// Convert fp16 (half precision) to fp32 (single precision)
// Usage: vec4f result = convert_fp16_to_fp32(fp16_vec);
// NEON: Single vcvt_f32_f16 instruction (3 cycles)
template<size_t N, typename Backend = auto_backend>
inline vec<float, N, Backend> convert_fp16_to_fp32(const vec<fp16_t, N, Backend>& v) {
    using actual_backend = typename vec<fp16_t, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, fp16_t, N>;
    auto result_reg = ops::convert_to_fp32(v.reg());
    return vec<float, N, Backend>(result_reg);
}

// Convert fp32 (single precision) to fp16 (half precision)
// Usage: vec<fp16_t, 4> result = convert_fp32_to_fp16(fp32_vec);
// NEON: Single vcvt_f16_f32 instruction (3 cycles)
template<size_t N, typename Backend = auto_backend>
inline vec<fp16_t, N, Backend> convert_fp32_to_fp16(const vec<float, N, Backend>& v) {
    using actual_backend = typename vec<float, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, float, N>;
    auto result_reg = ops::convert_to_fp16(v.reg());
    return vec<fp16_t, N, Backend>(result_reg);
}

//-----------------------------------------------------------------------------
// Integer-Float Conversions: int32 <-> float32
//-----------------------------------------------------------------------------

// Convert integer to float
// Usage: vec4f result = convert_to_float(int32_vec);
// NEON: Single vcvtq_f32_s32 or vcvtq_f32_u32 instruction (3 cycles)
template<typename IntT, size_t N, typename Backend = auto_backend>
inline vec<float, N, Backend> convert_to_float(const vec<IntT, N, Backend>& v) {
    static_assert(std::is_integral<IntT>::value, "Source type must be integer");
    using actual_backend = typename vec<IntT, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, IntT, N>;
    auto result_reg = ops::convert_to_float(v.reg());
    return vec<float, N, Backend>(result_reg);
}

// Convert float to integer with rounding (round to nearest)
// Usage: vec<int32_t, 4> result = convert_to_int<int32_t>(float_vec);
// NEON: Single vcvtq_s32_f32 or vcvtq_u32_f32 instruction (3 cycles)
template<typename IntT, size_t N, typename Backend = auto_backend>
inline vec<IntT, N, Backend> convert_to_int(const vec<float, N, Backend>& v) {
    static_assert(std::is_integral<IntT>::value, "Target type must be integer");
    using actual_backend = typename vec<float, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, float, N>;
    auto result_reg = ops::template convert_to_int<IntT>(v.reg());
    return vec<IntT, N, Backend>(result_reg);
}

//=============================================================================
// Phase 2: Integer Width Conversions (整数宽度转换)
//=============================================================================
//   1. Widening conversions: int8→int16, int16→int32, etc.
//   2. Narrowing conversions: int32→int16, int16→int8, etc.
//   3. Saturating narrowing: prevent overflow

//-----------------------------------------------------------------------------
// Widening Conversions (扩宽转换)
//-----------------------------------------------------------------------------
// Automatically widen integer types to twice their size
// NEON: vmovl_s8, vmovl_u8, vmovl_s16, vmovl_u16, vmovl_s32, vmovl_u32

// Helper trait: determine widened type
template<typename T> struct widen_type;
template<> struct widen_type<int8_t>   { using type = int16_t; };
template<> struct widen_type<uint8_t>  { using type = uint16_t; };
template<> struct widen_type<int16_t>  { using type = int32_t; };
template<> struct widen_type<uint16_t> { using type = uint32_t; };
template<> struct widen_type<int32_t>  { using type = int64_t; };
template<> struct widen_type<uint32_t> { using type = uint64_t; };

// Widen integer vector to twice the element size
// Usage: vec<int16_t, 8> wide = convert_widen(vec<int8_t, 8>)
// NEON: Single vmovl instruction (1 cycle)
template<typename T, size_t N, typename Backend = auto_backend>
inline vec<typename widen_type<T>::type, N, Backend> convert_widen(const vec<T, N, Backend>& v) {
    static_assert(std::is_integral<T>::value, "Source type must be integer");
    static_assert(sizeof(T) < 8, "Cannot widen 64-bit integers");

    using actual_backend = typename vec<T, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, T, N>;
    using target_type = typename widen_type<T>::type;

    auto result_reg = ops::convert_widen(v.reg());
    return vec<target_type, N, Backend>(result_reg);
}

//-----------------------------------------------------------------------------
// Narrowing Conversions (收窄转换)
//-----------------------------------------------------------------------------
// Narrow integer types to half their size
// NEON: vmovn_s16, vmovn_u16, vmovn_s32, vmovn_u32

// Helper trait: determine narrowed type
template<typename T> struct narrow_type;
template<> struct narrow_type<int16_t>  { using type = int8_t; };
template<> struct narrow_type<uint16_t> { using type = uint8_t; };
template<> struct narrow_type<int32_t>  { using type = int16_t; };
template<> struct narrow_type<uint32_t> { using type = uint16_t; };
template<> struct narrow_type<int64_t>  { using type = int32_t; };
template<> struct narrow_type<uint64_t> { using type = uint32_t; };

// Narrow integer vector to half the element size (may overflow)
// Usage: vec<int8_t, 8> narrow = convert_narrow(vec<int16_t, 8>)
// NEON: Single vmovn instruction (1 cycle)
template<typename T, size_t N, typename Backend = auto_backend>
inline vec<typename narrow_type<T>::type, N, Backend> convert_narrow(const vec<T, N, Backend>& v) {
    static_assert(std::is_integral<T>::value, "Source type must be integer");
    static_assert(sizeof(T) > 1, "Cannot narrow 8-bit integers");

    using actual_backend = typename vec<T, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, T, N>;
    using target_type = typename narrow_type<T>::type;

    auto result_reg = ops::convert_narrow(v.reg());
    return vec<target_type, N, Backend>(result_reg);
}

// Narrow with saturation (prevents overflow)
// Usage: vec<uint8_t, 8> narrow = convert_narrow_sat(vec<uint16_t, 8>)
// NEON: Single vqmovn or vqmovun instruction (1 cycle)
template<typename T, size_t N, typename Backend = auto_backend>
inline vec<typename narrow_type<T>::type, N, Backend> convert_narrow_sat(const vec<T, N, Backend>& v) {
    static_assert(std::is_integral<T>::value, "Source type must be integer");
    static_assert(sizeof(T) > 1, "Cannot narrow 8-bit integers");

    using actual_backend = typename vec<T, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, T, N>;
    using target_type = typename narrow_type<T>::type;

    auto result_reg = ops::convert_narrow_sat(v.reg());
    return vec<target_type, N, Backend>(result_reg);
}

//=============================================================================
// Phase 3: Unsigned to Signed Conversions (无符号 → 有符号转换)
//=============================================================================
//   1. Same-width conversions: uint32→int32, uint16→int16, uint8→int8
//   2. Saturating narrowing: uint16→int8 (with saturation)

//-----------------------------------------------------------------------------
// Same-Width Unsigned to Signed (同宽度转换)
//-----------------------------------------------------------------------------
// Convert unsigned to signed (same width, reinterpret bits)
// Usage: vec<int32_t, 4> s = convert_to_signed(vec<uint32_t, 4>)
// NEON: vreinterpretq_s32_u32, vreinterpretq_s16_u16, etc. (0 cycle)

template<typename T, size_t N, typename Backend = auto_backend>
inline typename std::enable_if<
    std::is_unsigned<T>::value,
    vec<typename std::make_signed<T>::type, N, Backend>
>::type
convert_to_signed(const vec<T, N, Backend>& v) {
    using actual_backend = typename vec<T, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, T, N>;
    using target_type = typename std::make_signed<T>::type;

    auto result_reg = ops::convert_to_signed(v.reg());
    return vec<target_type, N, Backend>(result_reg);
}

//-----------------------------------------------------------------------------
// Saturating Narrowing to Signed (饱和窄化转换)
//-----------------------------------------------------------------------------
// Convert unsigned to signed with narrowing and saturation
// Values > max_signed → max_signed (e.g., uint16[300] → int8[127])
// Usage: vec<int8_t, 8> s = convert_to_signed_sat(vec<uint16_t, 8>)
// NEON: vqmovn_u16 + reinterpret (1 cycle)

template<typename T, size_t N, typename Backend = auto_backend>
inline typename std::enable_if<
    (std::is_unsigned<T>::value && sizeof(T) > 1),
    vec<typename std::make_signed<typename narrow_type<T>::type>::type, N, Backend>
>::type
convert_to_signed_sat(const vec<T, N, Backend>& v) {
    using actual_backend = typename vec<T, N, Backend>::backend_type;
    using ops = backend_ops<actual_backend, T, N>;
    using unsigned_narrow = typename narrow_type<T>::type;
    using target_type = typename std::make_signed<unsigned_narrow>::type;

    auto result_reg = ops::convert_to_signed_sat(v.reg());
    return vec<target_type, N, Backend>(result_reg);
}

} // namespace tiny_simd

#endif // TINY_SIMD_BASE_HPP
