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
// Backend Tags - Backend type markers
//=============================================================================

struct scalar_backend {};  // Scalar backend (fallback)
struct neon_backend {};    // ARM NEON backend
struct sse_backend {};     // x86 SSE backend (future)
struct avx_backend {};     // x86 AVX backend (future)
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
        // int32/uint32: N=4 has NEON advantage
        ((std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value) && N == 4) ||
        // int16/uint16: N=8 has NEON advantage
        ((std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) && N == 8) ||
        // int8/uint8: N=16 has NEON advantage
        ((std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) && N == 16) ||
        // fp16: N=8 has NEON advantage (if supported)
        (std::is_same<T, fp16_t>::value && N == 8);
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
        scalar_backend
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
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type");

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

//=============================================================================
// Widening Operations (防溢出)
//=============================================================================

// Note: These generic implementations should be overridden by backend-specific optimizations

// uint8 widening add -> uint16
template<size_t N, typename Backend = auto_backend>
inline vec<uint16_t, N, Backend> add_wide(const vec<uint8_t, N, Backend>& a, const vec<uint8_t, N, Backend>& b) {
    alignas(32) uint8_t temp_a[N], temp_b[N];
    alignas(32) uint16_t result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint16_t>(temp_a[i]) + static_cast<uint16_t>(temp_b[i]);
    }
    return vec<uint16_t, N, Backend>(result);
}

// uint8 widening mul -> uint16
template<size_t N, typename Backend = auto_backend>
inline vec<uint16_t, N, Backend> mul_wide(const vec<uint8_t, N, Backend>& a, const vec<uint8_t, N, Backend>& b) {
    alignas(32) uint8_t temp_a[N], temp_b[N];
    alignas(32) uint16_t result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint16_t>(temp_a[i]) * static_cast<uint16_t>(temp_b[i]);
    }
    return vec<uint16_t, N, Backend>(result);
}

// uint16 widening add -> uint32
template<size_t N, typename Backend = auto_backend>
inline vec<uint32_t, N, Backend> add_wide(const vec<uint16_t, N, Backend>& a, const vec<uint16_t, N, Backend>& b) {
    alignas(32) uint16_t temp_a[N], temp_b[N];
    alignas(32) uint32_t result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint32_t>(temp_a[i]) + static_cast<uint32_t>(temp_b[i]);
    }
    return vec<uint32_t, N, Backend>(result);
}

// uint16 widening mul -> uint32
template<size_t N, typename Backend = auto_backend>
inline vec<uint32_t, N, Backend> mul_wide(const vec<uint16_t, N, Backend>& a, const vec<uint16_t, N, Backend>& b) {
    alignas(32) uint16_t temp_a[N], temp_b[N];
    alignas(32) uint32_t result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint32_t>(temp_a[i]) * static_cast<uint32_t>(temp_b[i]);
    }
    return vec<uint32_t, N, Backend>(result);
}

// int8 widening add -> int16
template<size_t N, typename Backend = auto_backend>
inline vec<int16_t, N, Backend> add_wide(const vec<int8_t, N, Backend>& a, const vec<int8_t, N, Backend>& b) {
    alignas(32) int8_t temp_a[N], temp_b[N];
    alignas(32) int16_t result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<int16_t>(temp_a[i]) + static_cast<int16_t>(temp_b[i]);
    }
    return vec<int16_t, N, Backend>(result);
}

// int8 widening mul -> int16
template<size_t N, typename Backend = auto_backend>
inline vec<int16_t, N, Backend> mul_wide(const vec<int8_t, N, Backend>& a, const vec<int8_t, N, Backend>& b) {
    alignas(32) int8_t temp_a[N], temp_b[N];
    alignas(32) int16_t result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<int16_t>(temp_a[i]) * static_cast<int16_t>(temp_b[i]);
    }
    return vec<int16_t, N, Backend>(result);
}

//=============================================================================
// Saturating Operations (饱和运算防溢出)
//=============================================================================

// uint8 saturating add
template<size_t N, typename Backend = auto_backend>
inline vec<uint8_t, N, Backend> add_sat(const vec<uint8_t, N, Backend>& a, const vec<uint8_t, N, Backend>& b) {
    alignas(32) uint8_t temp_a[N], temp_b[N], result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        uint16_t sum = static_cast<uint16_t>(temp_a[i]) + static_cast<uint16_t>(temp_b[i]);
        result[i] = (sum > 255) ? 255 : static_cast<uint8_t>(sum);
    }
    return vec<uint8_t, N, Backend>(result);
}

// uint8 saturating sub
template<size_t N, typename Backend = auto_backend>
inline vec<uint8_t, N, Backend> sub_sat(const vec<uint8_t, N, Backend>& a, const vec<uint8_t, N, Backend>& b) {
    alignas(32) uint8_t temp_a[N], temp_b[N], result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        result[i] = (temp_a[i] > temp_b[i]) ? (temp_a[i] - temp_b[i]) : 0;
    }
    return vec<uint8_t, N, Backend>(result);
}

// uint16 saturating add
template<size_t N, typename Backend = auto_backend>
inline vec<uint16_t, N, Backend> add_sat(const vec<uint16_t, N, Backend>& a, const vec<uint16_t, N, Backend>& b) {
    alignas(32) uint16_t temp_a[N], temp_b[N], result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        uint32_t sum = static_cast<uint32_t>(temp_a[i]) + static_cast<uint32_t>(temp_b[i]);
        result[i] = (sum > 65535) ? 65535 : static_cast<uint16_t>(sum);
    }
    return vec<uint16_t, N, Backend>(result);
}

// uint16 saturating sub
template<size_t N, typename Backend = auto_backend>
inline vec<uint16_t, N, Backend> sub_sat(const vec<uint16_t, N, Backend>& a, const vec<uint16_t, N, Backend>& b) {
    alignas(32) uint16_t temp_a[N], temp_b[N], result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        result[i] = (temp_a[i] > temp_b[i]) ? (temp_a[i] - temp_b[i]) : 0;
    }
    return vec<uint16_t, N, Backend>(result);
}

// int8 saturating add
template<size_t N, typename Backend = auto_backend>
inline vec<int8_t, N, Backend> add_sat(const vec<int8_t, N, Backend>& a, const vec<int8_t, N, Backend>& b) {
    alignas(32) int8_t temp_a[N], temp_b[N], result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        int16_t sum = static_cast<int16_t>(temp_a[i]) + static_cast<int16_t>(temp_b[i]);
        if (sum > 127) result[i] = 127;
        else if (sum < -128) result[i] = -128;
        else result[i] = static_cast<int8_t>(sum);
    }
    return vec<int8_t, N, Backend>(result);
}

// int8 saturating sub
template<size_t N, typename Backend = auto_backend>
inline vec<int8_t, N, Backend> sub_sat(const vec<int8_t, N, Backend>& a, const vec<int8_t, N, Backend>& b) {
    alignas(32) int8_t temp_a[N], temp_b[N], result[N];
    a.store(temp_a);
    b.store(temp_b);
    for (size_t i = 0; i < N; ++i) {
        int16_t diff = static_cast<int16_t>(temp_a[i]) - static_cast<int16_t>(temp_b[i]);
        if (diff > 127) result[i] = 127;
        else if (diff < -128) result[i] = -128;
        else result[i] = static_cast<int8_t>(diff);
    }
    return vec<int8_t, N, Backend>(result);
}

//=============================================================================
// Narrowing Saturation (窄化饱和)
//=============================================================================

// uint16 -> uint8 with saturation
template<size_t N, typename Backend = auto_backend>
inline vec<uint8_t, N, Backend> narrow_sat(const vec<uint16_t, N, Backend>& wide_result) {
    alignas(32) uint16_t temp[N];
    alignas(32) uint8_t result[N];
    wide_result.store(temp);
    for (size_t i = 0; i < N; ++i) {
        result[i] = (temp[i] > 255) ? 255 : static_cast<uint8_t>(temp[i]);
    }
    return vec<uint8_t, N, Backend>(result);
}

// uint32 -> uint16 with saturation
template<size_t N, typename Backend = auto_backend>
inline vec<uint16_t, N, Backend> narrow_sat(const vec<uint32_t, N, Backend>& wide_result) {
    alignas(32) uint32_t temp[N];
    alignas(32) uint16_t result[N];
    wide_result.store(temp);
    for (size_t i = 0; i < N; ++i) {
        result[i] = (temp[i] > 65535) ? 65535 : static_cast<uint16_t>(temp[i]);
    }
    return vec<uint16_t, N, Backend>(result);
}

// int16 -> int8 with saturation
template<size_t N, typename Backend = auto_backend>
inline vec<int8_t, N, Backend> narrow_sat(const vec<int16_t, N, Backend>& wide_result) {
    alignas(32) int16_t temp[N];
    alignas(32) int8_t result[N];
    wide_result.store(temp);
    for (size_t i = 0; i < N; ++i) {
        if (temp[i] > 127) result[i] = 127;
        else if (temp[i] < -128) result[i] = -128;
        else result[i] = static_cast<int8_t>(temp[i]);
    }
    return vec<int8_t, N, Backend>(result);
}

} // namespace tiny_simd

#endif // TINY_SIMD_BASE_HPP
