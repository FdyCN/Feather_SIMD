#ifndef TINY_SIMD_HPP
#define TINY_SIMD_HPP

#include <cstddef>
#include <type_traits>
#include <cmath>
#include <cassert>
#include <array>
#include <initializer_list>
#include <cstdint>

// Half precision float type definition
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE)
    // Native ARM FP16 support
    using fp16_t = __fp16;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    // x86 with GCC - use uint16_t with conversion functions
    using fp16_t = uint16_t;
#else
    // Fallback - emulate with uint16_t
    using fp16_t = uint16_t;
#endif

// 标准库兼容性检查
#if __cplusplus < 201103L
    #error "Tiny SIMD Engine requires C++11 or later"
#endif

//=============================================================================
// 平台检测和SIMD指令集配置
//=============================================================================

// SIMD指令集包含文件
#ifdef TINY_SIMD_ARM_NEON
    #include <arm_neon.h>
#endif

#ifdef TINY_SIMD_X86_SSE
    #include <immintrin.h>
#endif

namespace tiny_simd {
namespace config {

// 平台检测宏定义
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

// 向量长度配置
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

// 对齐要求
#if defined(TINY_SIMD_ARM_NEON) || defined(TINY_SIMD_X86_SSE)
static constexpr size_t simd_alignment = 16;
#elif defined(TINY_SIMD_X86_AVX)
static constexpr size_t simd_alignment = 32;
#else
static constexpr size_t simd_alignment = sizeof(void*);
#endif

} // namespace config

//=============================================================================
// 核心向量类 (纯标量实现)
//=============================================================================

template<typename T, size_t N>
class simd_vector {
private:
    static_assert(N > 0, "Vector size must be positive");
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic type");

    // 使用适当的对齐方式
    alignas(config::simd_alignment) T data_[N];

public:
    using value_type = T;
    using size_type = size_t;
    static constexpr size_type size_value = N;

#if defined(TINY_SIMD_ARM_NEON)
    static constexpr bool is_simd_optimized = (N == 2 && std::is_same<T, float>::value) ||
                                             (N == 4 && std::is_same<T, float>::value) ||
                                             (N == 4 && std::is_same<T, int32_t>::value) ||
                                             (N == 4 && std::is_same<T, uint32_t>::value) ||
                                             (N == 8 && std::is_same<T, int16_t>::value) ||
                                             (N == 8 && std::is_same<T, uint16_t>::value) ||
                                             (N == 16 && std::is_same<T, int8_t>::value) ||
                                             (N == 16 && std::is_same<T, uint8_t>::value) ||
                                             (N == 8 && std::is_same<T, fp16_t>::value);
#else
    static constexpr bool is_simd_optimized = false;
#endif

    //=============================================================================
    // 构造函数
    //=============================================================================

    // 默认构造函数
    simd_vector() = default;

    // 标量构造函数
    explicit simd_vector(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] = scalar;
        }
    }

    // 从initializer_list构造
    simd_vector(std::initializer_list<T> init) {
        assert(init.size() == N);
        auto it = init.begin();
        for (size_t i = 0; i < N; ++i) {
            data_[i] = *it++;
        }
    }

    // 从指针构造 - 优先级高于variadic template
    simd_vector(const T* ptr) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] = ptr[i];
        }
    }

    // 从对齐指针构造
    static simd_vector load_aligned(const T* ptr) {
        simd_vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data_[i] = ptr[i];
        }
        return result;
    }

    //=============================================================================
    // 数据访问
    //=============================================================================

    T& operator[](size_t i) {
        assert(i < N);
        return data_[i];
    }

    const T& operator[](size_t i) const {
        assert(i < N);
        return data_[i];
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

    constexpr size_t size() const { return N; }

    // 存储到指针
    void store(T* ptr) const {
        for (size_t i = 0; i < N; ++i) {
            ptr[i] = data_[i];
        }
    }

    // 存储到对齐指针
    void store_aligned(T* ptr) const {
        for (size_t i = 0; i < N; ++i) {
            ptr[i] = data_[i];
        }
    }

    //=============================================================================
    // 算术运算符
    //=============================================================================

    simd_vector& operator+=(const simd_vector& other) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    simd_vector& operator-=(const simd_vector& other) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    simd_vector& operator*=(const simd_vector& other) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] *= other.data_[i];
        }
        return *this;
    }

    simd_vector& operator/=(const simd_vector& other) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] /= other.data_[i];
        }
        return *this;
    }

    // 标量运算
    simd_vector& operator+=(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] += scalar;
        }
        return *this;
    }

    simd_vector& operator-=(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] -= scalar;
        }
        return *this;
    }

    simd_vector& operator*=(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    simd_vector& operator/=(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] /= scalar;
        }
        return *this;
    }

    // 一元运算符
    simd_vector operator-() const {
        simd_vector result;
        for (size_t i = 0; i < N; ++i) {
            result.data_[i] = -data_[i];
        }
        return result;
    }

    //=============================================================================
    // 比较运算符
    //=============================================================================

    bool operator==(const simd_vector& other) const {
        for (size_t i = 0; i < N; ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }

    bool operator!=(const simd_vector& other) const {
        return !(*this == other);
    }
};

//=============================================================================
// 二元运算符
//=============================================================================

template<typename T, size_t N>
inline simd_vector<T, N> operator+(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    simd_vector<T, N> result = a;
    result += b;
    return result;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator-(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    simd_vector<T, N> result = a;
    result -= b;
    return result;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator*(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    simd_vector<T, N> result = a;
    result *= b;
    return result;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator/(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    simd_vector<T, N> result = a;
    result /= b;
    return result;
}

// 标量运算
template<typename T, size_t N>
inline simd_vector<T, N> operator+(const simd_vector<T, N>& v, T scalar) {
    simd_vector<T, N> result = v;
    result += scalar;
    return result;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator+(T scalar, const simd_vector<T, N>& v) {
    return v + scalar;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator-(const simd_vector<T, N>& v, T scalar) {
    simd_vector<T, N> result = v;
    result -= scalar;
    return result;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator-(T scalar, const simd_vector<T, N>& v) {
    simd_vector<T, N> result(scalar);
    result -= v;
    return result;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator*(const simd_vector<T, N>& v, T scalar) {
    simd_vector<T, N> result = v;
    result *= scalar;
    return result;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator*(T scalar, const simd_vector<T, N>& v) {
    return v * scalar;
}

template<typename T, size_t N>
inline simd_vector<T, N> operator/(const simd_vector<T, N>& v, T scalar) {
    simd_vector<T, N> result = v;
    result /= scalar;
    return result;
}

//=============================================================================
// 常用类型别名
//=============================================================================

using vec2f = simd_vector<float, 2>;
using vec3f = simd_vector<float, 3>;
using vec4f = simd_vector<float, 4>;
using vec8f = simd_vector<float, 8>;

using vec2d = simd_vector<double, 2>;
using vec4d = simd_vector<double, 4>;

using vec4h = simd_vector<fp16_t, 4>;
using vec8h = simd_vector<fp16_t, 8>;
using vec16h = simd_vector<fp16_t, 16>;

using vec4i = simd_vector<int32_t, 4>;
using vec8i = simd_vector<int32_t, 8>;

using vec4ui = simd_vector<uint32_t, 4>;
using vec8ui = simd_vector<uint32_t, 8>;

using vec8s = simd_vector<int16_t, 8>;
using vec16s = simd_vector<int16_t, 16>;

using vec8us = simd_vector<uint16_t, 8>;
using vec16us = simd_vector<uint16_t, 16>;

using vec16b = simd_vector<int8_t, 16>;
using vec32b = simd_vector<int8_t, 32>;

using vec16ub = simd_vector<uint8_t, 16>;
using vec32ub = simd_vector<uint8_t, 32>;

//=============================================================================
// 加宽运算函数 (Widening Operations) - 防溢出
//=============================================================================

// uint8 加宽加法 - 返回 uint16
template<size_t N>
inline simd_vector<uint16_t, N> add_wide(const simd_vector<uint8_t, N>& a, const simd_vector<uint8_t, N>& b) {
    simd_vector<uint16_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint16_t>(a[i]) + static_cast<uint16_t>(b[i]);
    }
    return result;
}

// uint8 加宽乘法 - 返回 uint16
template<size_t N>
inline simd_vector<uint16_t, N> mul_wide(const simd_vector<uint8_t, N>& a, const simd_vector<uint8_t, N>& b) {
    simd_vector<uint16_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint16_t>(a[i]) * static_cast<uint16_t>(b[i]);
    }
    return result;
}

// uint16 加宽加法 - 返回 uint32
template<size_t N>
inline simd_vector<uint32_t, N> add_wide(const simd_vector<uint16_t, N>& a, const simd_vector<uint16_t, N>& b) {
    simd_vector<uint32_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint32_t>(a[i]) + static_cast<uint32_t>(b[i]);
    }
    return result;
}

// uint16 加宽乘法 - 返回 uint32
template<size_t N>
inline simd_vector<uint32_t, N> mul_wide(const simd_vector<uint16_t, N>& a, const simd_vector<uint16_t, N>& b) {
    simd_vector<uint32_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
    }
    return result;
}

// int8 加宽加法 - 返回 int16
template<size_t N>
inline simd_vector<int16_t, N> add_wide(const simd_vector<int8_t, N>& a, const simd_vector<int8_t, N>& b) {
    simd_vector<int16_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<int16_t>(a[i]) + static_cast<int16_t>(b[i]);
    }
    return result;
}

// int8 加宽乘法 - 返回 int16
template<size_t N>
inline simd_vector<int16_t, N> mul_wide(const simd_vector<int8_t, N>& a, const simd_vector<int8_t, N>& b) {
    simd_vector<int16_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<int16_t>(a[i]) * static_cast<int16_t>(b[i]);
    }
    return result;
}

//=============================================================================
// 饱和运算函数 (Saturating Operations) - 防溢出
//=============================================================================

// 饱和加法 - uint8
template<size_t N>
inline simd_vector<uint8_t, N> add_sat(const simd_vector<uint8_t, N>& a, const simd_vector<uint8_t, N>& b) {
    simd_vector<uint8_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        uint16_t sum = static_cast<uint16_t>(a[i]) + static_cast<uint16_t>(b[i]);
        result[i] = (sum > 255) ? 255 : static_cast<uint8_t>(sum);
    }
    return result;
}

// 饱和减法 - uint8
template<size_t N>
inline simd_vector<uint8_t, N> sub_sat(const simd_vector<uint8_t, N>& a, const simd_vector<uint8_t, N>& b) {
    simd_vector<uint8_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = (a[i] > b[i]) ? (a[i] - b[i]) : 0;
    }
    return result;
}

// 饱和加法 - uint16
template<size_t N>
inline simd_vector<uint16_t, N> add_sat(const simd_vector<uint16_t, N>& a, const simd_vector<uint16_t, N>& b) {
    simd_vector<uint16_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        uint32_t sum = static_cast<uint32_t>(a[i]) + static_cast<uint32_t>(b[i]);
        result[i] = (sum > 65535) ? 65535 : static_cast<uint16_t>(sum);
    }
    return result;
}

// 饱和减法 - uint16
template<size_t N>
inline simd_vector<uint16_t, N> sub_sat(const simd_vector<uint16_t, N>& a, const simd_vector<uint16_t, N>& b) {
    simd_vector<uint16_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = (a[i] > b[i]) ? (a[i] - b[i]) : 0;
    }
    return result;
}

// 饱和加法 - int8
template<size_t N>
inline simd_vector<int8_t, N> add_sat(const simd_vector<int8_t, N>& a, const simd_vector<int8_t, N>& b) {
    simd_vector<int8_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        int16_t sum = static_cast<int16_t>(a[i]) + static_cast<int16_t>(b[i]);
        if (sum > 127) result[i] = 127;
        else if (sum < -128) result[i] = -128;
        else result[i] = static_cast<int8_t>(sum);
    }
    return result;
}

// 饱和减法 - int8
template<size_t N>
inline simd_vector<int8_t, N> sub_sat(const simd_vector<int8_t, N>& a, const simd_vector<int8_t, N>& b) {
    simd_vector<int8_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        int16_t diff = static_cast<int16_t>(a[i]) - static_cast<int16_t>(b[i]);
        if (diff > 127) result[i] = 127;
        else if (diff < -128) result[i] = -128;
        else result[i] = static_cast<int8_t>(diff);
    }
    return result;
}

//=============================================================================
// 窄化饱和函数 (Narrowing Saturation) - 从宽类型饱和转换回窄类型
//=============================================================================

// 从加宽结果饱和转换回原类型
template<size_t N>
inline simd_vector<uint8_t, N> narrow_sat(const simd_vector<uint16_t, N>& wide_result) {
    simd_vector<uint8_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = (wide_result[i] > 255) ? 255 : static_cast<uint8_t>(wide_result[i]);
    }
    return result;
}

template<size_t N>
inline simd_vector<uint16_t, N> narrow_sat(const simd_vector<uint32_t, N>& wide_result) {
    simd_vector<uint16_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = (wide_result[i] > 65535) ? 65535 : static_cast<uint16_t>(wide_result[i]);
    }
    return result;
}

template<size_t N>
inline simd_vector<int8_t, N> narrow_sat(const simd_vector<int16_t, N>& wide_result) {
    simd_vector<int8_t, N> result;
    for (size_t i = 0; i < N; ++i) {
        if (wide_result[i] > 127) result[i] = 127;
        else if (wide_result[i] < -128) result[i] = -128;
        else result[i] = static_cast<int8_t>(wide_result[i]);
    }
    return result;
}

// 点积
template<typename T, size_t N>
inline T dot(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    T result = T{0};
    for (size_t i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// 向量长度平方
template<typename T, size_t N>
inline T length_squared(const simd_vector<T, N>& v) {
    return dot(v, v);
}

// 向量长度
template<typename T, size_t N>
inline T length(const simd_vector<T, N>& v) {
    return std::sqrt(length_squared(v));
}

// 归一化
template<typename T, size_t N>
inline simd_vector<T, N> normalize(const simd_vector<T, N>& v) {
    T len = length(v);
    if (len == T{0}) return v;
    return v / len;
}

// 向量间距离
template<typename T, size_t N>
inline T distance(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    return length(a - b);
}

// 向量间距离平方
template<typename T, size_t N>
inline T distance_squared(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    return length_squared(a - b);
}

// 线性插值
template<typename T, size_t N>
inline simd_vector<T, N> lerp(const simd_vector<T, N>& a, const simd_vector<T, N>& b, T t) {
    return a + (b - a) * t;
}

// 向量投影 (a在b上的投影)
template<typename T, size_t N>
inline simd_vector<T, N> project(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    T b_length_sq = length_squared(b);
    if (b_length_sq == T{0}) return simd_vector<T, N>(T{0});
    return b * (dot(a, b) / b_length_sq);
}

// 向量反射
template<typename T, size_t N>
inline simd_vector<T, N> reflect(const simd_vector<T, N>& v, const simd_vector<T, N>& n) {
    return v - n * (T{2} * dot(v, n));
}

// 3D向量叉积
template<typename T>
inline simd_vector<T, 3> cross(const simd_vector<T, 3>& a, const simd_vector<T, 3>& b) {
    return simd_vector<T, 3>{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// 最小值
template<typename T, size_t N>
inline simd_vector<T, N> min(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    simd_vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::min(a[i], b[i]);
    }
    return result;
}

// 最大值
template<typename T, size_t N>
inline simd_vector<T, N> max(const simd_vector<T, N>& a, const simd_vector<T, N>& b) {
    simd_vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::max(a[i], b[i]);
    }
    return result;
}

// 限制在范围内
template<typename T, size_t N>
inline simd_vector<T, N> clamp(const simd_vector<T, N>& v, const simd_vector<T, N>& min_val, const simd_vector<T, N>& max_val) {
    return min(max(v, min_val), max_val);
}

template<typename T, size_t N>
inline simd_vector<T, N> clamp(const simd_vector<T, N>& v, T min_val, T max_val) {
    return clamp(v, simd_vector<T, N>(min_val), simd_vector<T, N>(max_val));
}

// 绝对值
template<typename T, size_t N>
inline simd_vector<T, N> abs(const simd_vector<T, N>& v) {
    simd_vector<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::abs(v[i]);
    }
    return result;
}

} // namespace tiny_simd

//=============================================================================
// NEON 特化版本
//=============================================================================

#ifdef TINY_SIMD_ARM_NEON

namespace tiny_simd {

//=============================================================================
// float32x2 特化版本 (vec2f)
//=============================================================================

template<>
class simd_vector<float, 2> {
private:
    union {
        float32x2_t neon_data;
        alignas(8) float data_[2];
    };

public:
    using value_type = float;
    using size_type = size_t;
    static constexpr size_type size_value = 2;
    static constexpr bool is_simd_optimized = true;

    //=============================================================================
    // 构造函数
    //=============================================================================

    simd_vector() : neon_data(vdup_n_f32(0.0f)) {}

    explicit simd_vector(float scalar) : neon_data(vdup_n_f32(scalar)) {}

    simd_vector(std::initializer_list<float> init) {
        assert(init.size() == 2);
        auto it = init.begin();
        data_[0] = *it++;
        data_[1] = *it++;
        neon_data = vld1_f32(data_);
    }

    simd_vector(const float* ptr) : neon_data(vld1_f32(ptr)) {}

    simd_vector(float32x2_t neon_vec) : neon_data(neon_vec) {}

    static simd_vector load_aligned(const float* ptr) {
        return simd_vector(vld1_f32(ptr));
    }

    //=============================================================================
    // 数据访问
    //=============================================================================

    float& operator[](size_t i) {
        assert(i < 2);
        vst1_f32(data_, neon_data);
        return data_[i];
    }

    const float& operator[](size_t i) const {
        assert(i < 2);
        float* mutable_data = const_cast<float*>(data_);
        vst1_f32(mutable_data, neon_data);
        return data_[i];
    }

    float* data() {
        vst1_f32(data_, neon_data);
        return data_;
    }

    const float* data() const {
        float* mutable_data = const_cast<float*>(data_);
        vst1_f32(mutable_data, neon_data);
        return data_;
    }

    constexpr size_t size() const { return 2; }

    void store(float* ptr) const {
        vst1_f32(ptr, neon_data);
    }

    void store_aligned(float* ptr) const {
        vst1_f32(ptr, neon_data);
    }

    //=============================================================================
    // 算术运算符
    //=============================================================================

    simd_vector& operator+=(const simd_vector& other) {
        neon_data = vadd_f32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator-=(const simd_vector& other) {
        neon_data = vsub_f32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator*=(const simd_vector& other) {
        neon_data = vmul_f32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator/=(const simd_vector& other) {
        // 注意：float32x2 没有除法指令，使用乘法的倒数
        float32x2_t reciprocal = vrecpe_f32(other.neon_data);
        reciprocal = vmul_f32(vrecps_f32(other.neon_data, reciprocal), reciprocal);
        neon_data = vmul_f32(neon_data, reciprocal);
        return *this;
    }

    simd_vector& operator+=(float scalar) {
        neon_data = vadd_f32(neon_data, vdup_n_f32(scalar));
        return *this;
    }

    simd_vector& operator-=(float scalar) {
        neon_data = vsub_f32(neon_data, vdup_n_f32(scalar));
        return *this;
    }

    simd_vector& operator*=(float scalar) {
        neon_data = vmul_f32(neon_data, vdup_n_f32(scalar));
        return *this;
    }

    simd_vector& operator/=(float scalar) {
        float32x2_t scalar_vec = vdup_n_f32(scalar);
        float32x2_t reciprocal = vrecpe_f32(scalar_vec);
        reciprocal = vmul_f32(vrecps_f32(scalar_vec, reciprocal), reciprocal);
        neon_data = vmul_f32(neon_data, reciprocal);
        return *this;
    }

    simd_vector operator-() const {
        return simd_vector(vneg_f32(neon_data));
    }

    //=============================================================================
    // 比较运算符
    //=============================================================================

    bool operator==(const simd_vector& other) const {
        uint32x2_t result = vceq_f32(neon_data, other.neon_data);
        // 检查两个元素都相等
        return vget_lane_u32(result, 0) == 0xFFFFFFFF && vget_lane_u32(result, 1) == 0xFFFFFFFF;
    }

    bool operator!=(const simd_vector& other) const {
        return !(*this == other);
    }

    // 访问内部NEON数据
    float32x2_t neon() const { return neon_data; }
};

// 静态成员定义
constexpr bool simd_vector<float, 2>::is_simd_optimized;

//=============================================================================
// float32x4 特化版本 (vec4f)
//=============================================================================

template<>
class simd_vector<float, 4> {
private:
    union {
        float32x4_t neon_data;
        alignas(16) float data_[4];
    };

public:
    using value_type = float;
    using size_type = size_t;
    static constexpr size_type size_value = 4;
    static constexpr bool is_simd_optimized = true;

    //=============================================================================
    // 构造函数
    //=============================================================================

    simd_vector() : neon_data(vdupq_n_f32(0.0f)) {}

    explicit simd_vector(float scalar) : neon_data(vdupq_n_f32(scalar)) {}

    simd_vector(std::initializer_list<float> init) {
        assert(init.size() == 4);
        auto it = init.begin();
        data_[0] = *it++;
        data_[1] = *it++;
        data_[2] = *it++;
        data_[3] = *it++;
        neon_data = vld1q_f32(data_);
    }

    simd_vector(const float* ptr) : neon_data(vld1q_f32(ptr)) {}

    simd_vector(float32x4_t neon_vec) : neon_data(neon_vec) {}

    static simd_vector load_aligned(const float* ptr) {
        return simd_vector(vld1q_f32(ptr));
    }

    //=============================================================================
    // 数据访问
    //=============================================================================

    float& operator[](size_t i) {
        assert(i < 4);
        vst1q_f32(data_, neon_data);
        return data_[i];
    }

    const float& operator[](size_t i) const {
        assert(i < 4);
        float* mutable_data = const_cast<float*>(data_);
        vst1q_f32(mutable_data, neon_data);
        return data_[i];
    }

    float* data() {
        vst1q_f32(data_, neon_data);
        return data_;
    }

    const float* data() const {
        float* mutable_data = const_cast<float*>(data_);
        vst1q_f32(mutable_data, neon_data);
        return data_;
    }

    constexpr size_t size() const { return 4; }

    void store(float* ptr) const {
        vst1q_f32(ptr, neon_data);
    }

    void store_aligned(float* ptr) const {
        vst1q_f32(ptr, neon_data);
    }

    //=============================================================================
    // 算术运算符
    //=============================================================================

    simd_vector& operator+=(const simd_vector& other) {
        neon_data = vaddq_f32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator-=(const simd_vector& other) {
        neon_data = vsubq_f32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator*=(const simd_vector& other) {
        neon_data = vmulq_f32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator/=(const simd_vector& other) {
        float32x4_t reciprocal = vrecpeq_f32(other.neon_data);
        reciprocal = vmulq_f32(vrecpsq_f32(other.neon_data, reciprocal), reciprocal);
        neon_data = vmulq_f32(neon_data, reciprocal);
        return *this;
    }

    simd_vector& operator+=(float scalar) {
        neon_data = vaddq_f32(neon_data, vdupq_n_f32(scalar));
        return *this;
    }

    simd_vector& operator-=(float scalar) {
        neon_data = vsubq_f32(neon_data, vdupq_n_f32(scalar));
        return *this;
    }

    simd_vector& operator*=(float scalar) {
        neon_data = vmulq_f32(neon_data, vdupq_n_f32(scalar));
        return *this;
    }

    simd_vector& operator/=(float scalar) {
        float32x4_t scalar_vec = vdupq_n_f32(scalar);
        float32x4_t reciprocal = vrecpeq_f32(scalar_vec);
        reciprocal = vmulq_f32(vrecpsq_f32(scalar_vec, reciprocal), reciprocal);
        neon_data = vmulq_f32(neon_data, reciprocal);
        return *this;
    }

    simd_vector operator-() const {
        return simd_vector(vnegq_f32(neon_data));
    }

    //=============================================================================
    // 比较运算符
    //=============================================================================

    bool operator==(const simd_vector& other) const {
        uint32x4_t result = vceqq_f32(neon_data, other.neon_data);
        uint64x2_t result64 = vpaddlq_u32(result);
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    bool operator!=(const simd_vector& other) const {
        return !(*this == other);
    }

    // 访问内部NEON数据
    float32x4_t neon() const { return neon_data; }
};

//=============================================================================
// int32x4 特化版本 (vec4i)
//=============================================================================

template<>
class simd_vector<int32_t, 4> {
private:
    union {
        int32x4_t neon_data;
        alignas(16) int32_t data_[4];
    };

public:
    using value_type = int32_t;
    using size_type = size_t;
    static constexpr size_type size_value = 4;
    static constexpr bool is_simd_optimized = true;

    //=============================================================================
    // 构造函数
    //=============================================================================

    simd_vector() : neon_data(vdupq_n_s32(0)) {}

    explicit simd_vector(int32_t scalar) : neon_data(vdupq_n_s32(scalar)) {}

    simd_vector(std::initializer_list<int32_t> init) {
        assert(init.size() == 4);
        auto it = init.begin();
        data_[0] = *it++;
        data_[1] = *it++;
        data_[2] = *it++;
        data_[3] = *it++;
        neon_data = vld1q_s32(data_);
    }

    simd_vector(const int32_t* ptr) : neon_data(vld1q_s32(ptr)) {}

    simd_vector(int32x4_t neon_vec) : neon_data(neon_vec) {}

    static simd_vector load_aligned(const int32_t* ptr) {
        return simd_vector(vld1q_s32(ptr));
    }

    //=============================================================================
    // 数据访问
    //=============================================================================

    int32_t& operator[](size_t i) {
        assert(i < 4);
        vst1q_s32(data_, neon_data);
        return data_[i];
    }

    const int32_t& operator[](size_t i) const {
        assert(i < 4);
        int32_t* mutable_data = const_cast<int32_t*>(data_);
        vst1q_s32(mutable_data, neon_data);
        return data_[i];
    }

    int32_t* data() {
        vst1q_s32(data_, neon_data);
        return data_;
    }

    const int32_t* data() const {
        int32_t* mutable_data = const_cast<int32_t*>(data_);
        vst1q_s32(mutable_data, neon_data);
        return data_;
    }

    constexpr size_t size() const { return 4; }

    void store(int32_t* ptr) const {
        vst1q_s32(ptr, neon_data);
    }

    void store_aligned(int32_t* ptr) const {
        vst1q_s32(ptr, neon_data);
    }

    //=============================================================================
    // 算术运算符
    //=============================================================================

    simd_vector& operator+=(const simd_vector& other) {
        neon_data = vaddq_s32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator-=(const simd_vector& other) {
        neon_data = vsubq_s32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator*=(const simd_vector& other) {
        neon_data = vmulq_s32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator+=(int32_t scalar) {
        neon_data = vaddq_s32(neon_data, vdupq_n_s32(scalar));
        return *this;
    }

    simd_vector& operator-=(int32_t scalar) {
        neon_data = vsubq_s32(neon_data, vdupq_n_s32(scalar));
        return *this;
    }

    simd_vector& operator*=(int32_t scalar) {
        neon_data = vmulq_s32(neon_data, vdupq_n_s32(scalar));
        return *this;
    }

    simd_vector operator-() const {
        return simd_vector(vnegq_s32(neon_data));
    }

    //=============================================================================
    // 比较运算符
    //=============================================================================

    bool operator==(const simd_vector& other) const {
        uint32x4_t result = vceqq_s32(neon_data, other.neon_data);
        uint64x2_t result64 = vpaddlq_u32(result);
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    bool operator!=(const simd_vector& other) const {
        return !(*this == other);
    }

    // 访问内部NEON数据
    int32x4_t neon() const { return neon_data; }
};

//=============================================================================
// uint32x4 特化版本 (vec4ui)
//=============================================================================

template<>
class simd_vector<uint32_t, 4> {
private:
    union {
        uint32x4_t neon_data;
        alignas(16) uint32_t data_[4];
    };

public:
    using value_type = uint32_t;
    using size_type = size_t;
    static constexpr size_type size_value = 4;
    static constexpr bool is_simd_optimized = true;

    //=============================================================================
    // 构造函数
    //=============================================================================

    simd_vector() : neon_data(vdupq_n_u32(0)) {}

    explicit simd_vector(uint32_t scalar) : neon_data(vdupq_n_u32(scalar)) {}

    simd_vector(std::initializer_list<uint32_t> init) {
        assert(init.size() == 4);
        auto it = init.begin();
        data_[0] = *it++;
        data_[1] = *it++;
        data_[2] = *it++;
        data_[3] = *it++;
        neon_data = vld1q_u32(data_);
    }

    simd_vector(const uint32_t* ptr) : neon_data(vld1q_u32(ptr)) {}

    simd_vector(uint32x4_t neon_vec) : neon_data(neon_vec) {}

    static simd_vector load_aligned(const uint32_t* ptr) {
        return simd_vector(vld1q_u32(ptr));
    }

    //=============================================================================
    // 数据访问
    //=============================================================================

    uint32_t& operator[](size_t i) {
        assert(i < 4);
        vst1q_u32(data_, neon_data);
        return data_[i];
    }

    const uint32_t& operator[](size_t i) const {
        assert(i < 4);
        uint32_t* mutable_data = const_cast<uint32_t*>(data_);
        vst1q_u32(mutable_data, neon_data);
        return data_[i];
    }

    uint32_t* data() {
        vst1q_u32(data_, neon_data);
        return data_;
    }

    const uint32_t* data() const {
        uint32_t* mutable_data = const_cast<uint32_t*>(data_);
        vst1q_u32(mutable_data, neon_data);
        return data_;
    }

    constexpr size_t size() const { return 4; }

    void store(uint32_t* ptr) const {
        vst1q_u32(ptr, neon_data);
    }

    void store_aligned(uint32_t* ptr) const {
        vst1q_u32(ptr, neon_data);
    }

    //=============================================================================
    // 算术运算符
    //=============================================================================

    simd_vector& operator+=(const simd_vector& other) {
        neon_data = vaddq_u32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator-=(const simd_vector& other) {
        neon_data = vsubq_u32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator*=(const simd_vector& other) {
        neon_data = vmulq_u32(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator+=(uint32_t scalar) {
        neon_data = vaddq_u32(neon_data, vdupq_n_u32(scalar));
        return *this;
    }

    simd_vector& operator-=(uint32_t scalar) {
        neon_data = vsubq_u32(neon_data, vdupq_n_u32(scalar));
        return *this;
    }

    simd_vector& operator*=(uint32_t scalar) {
        neon_data = vmulq_u32(neon_data, vdupq_n_u32(scalar));
        return *this;
    }

    //=============================================================================
    // 比较运算符
    //=============================================================================

    bool operator==(const simd_vector& other) const {
        uint32x4_t result = vceqq_u32(neon_data, other.neon_data);
        uint64x2_t result64 = vpaddlq_u32(result);
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    bool operator!=(const simd_vector& other) const {
        return !(*this == other);
    }

    // 访问内部NEON数据
    uint32x4_t neon() const { return neon_data; }
};

//=============================================================================
// uint8x16 特化版本 (vec16ub)
//=============================================================================

template<>
class simd_vector<uint8_t, 16> {
private:
    union {
        uint8x16_t neon_data;
        alignas(16) uint8_t data_[16];
    };

public:
    using value_type = uint8_t;
    using size_type = size_t;
    static constexpr size_type size_value = 16;
    static constexpr bool is_simd_optimized = true;

    //=============================================================================
    // 构造函数
    //=============================================================================

    simd_vector() : neon_data(vdupq_n_u8(0)) {}

    explicit simd_vector(uint8_t scalar) : neon_data(vdupq_n_u8(scalar)) {}

    simd_vector(std::initializer_list<uint8_t> init) {
        assert(init.size() == 16);
        auto it = init.begin();
        for (size_t i = 0; i < 16; ++i) {
            data_[i] = *it++;
        }
        neon_data = vld1q_u8(data_);
    }

    simd_vector(const uint8_t* ptr) : neon_data(vld1q_u8(ptr)) {}

    simd_vector(uint8x16_t neon_vec) : neon_data(neon_vec) {}

    static simd_vector load_aligned(const uint8_t* ptr) {
        return simd_vector(vld1q_u8(ptr));
    }

    //=============================================================================
    // 数据访问
    //=============================================================================

    uint8_t& operator[](size_t i) {
        assert(i < 16);
        vst1q_u8(data_, neon_data);
        return data_[i];
    }

    const uint8_t& operator[](size_t i) const {
        assert(i < 16);
        uint8_t* mutable_data = const_cast<uint8_t*>(data_);
        vst1q_u8(mutable_data, neon_data);
        return data_[i];
    }

    uint8_t* data() {
        vst1q_u8(data_, neon_data);
        return data_;
    }

    const uint8_t* data() const {
        uint8_t* mutable_data = const_cast<uint8_t*>(data_);
        vst1q_u8(mutable_data, neon_data);
        return data_;
    }

    constexpr size_t size() const { return 16; }

    void store(uint8_t* ptr) const {
        vst1q_u8(ptr, neon_data);
    }

    void store_aligned(uint8_t* ptr) const {
        vst1q_u8(ptr, neon_data);
    }

    //=============================================================================
    // 算术运算符
    //=============================================================================

    simd_vector& operator+=(const simd_vector& other) {
        neon_data = vaddq_u8(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator-=(const simd_vector& other) {
        neon_data = vsubq_u8(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator*=(const simd_vector& other) {
        neon_data = vmulq_u8(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator+=(uint8_t scalar) {
        neon_data = vaddq_u8(neon_data, vdupq_n_u8(scalar));
        return *this;
    }

    simd_vector& operator-=(uint8_t scalar) {
        neon_data = vsubq_u8(neon_data, vdupq_n_u8(scalar));
        return *this;
    }

    simd_vector& operator*=(uint8_t scalar) {
        neon_data = vmulq_u8(neon_data, vdupq_n_u8(scalar));
        return *this;
    }

    //=============================================================================
    // 比较运算符
    //=============================================================================

    bool operator==(const simd_vector& other) const {
        uint8x16_t result = vceqq_u8(neon_data, other.neon_data);
        // 使用正确的水平求和方式
        uint16x8_t result16 = vpaddlq_u8(result);  // uint8x16 → uint16x8
        uint32x4_t result32 = vpaddlq_u16(result16);  // uint16x8 → uint32x4
        uint64x2_t result64 = vpaddlq_u32(result32);  // uint32x4 → uint64x2
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    bool operator!=(const simd_vector& other) const {
        return !(*this == other);
    }

    // 访问内部NEON数据
    uint8x16_t neon() const { return neon_data; }
};

//=============================================================================
// uint16x8 特化版本 (vec8us)
//=============================================================================

template<>
class simd_vector<uint16_t, 8> {
private:
    union {
        uint16x8_t neon_data;
        alignas(16) uint16_t data_[8];
    };

public:
    using value_type = uint16_t;
    using size_type = size_t;
    static constexpr size_type size_value = 8;
    static constexpr bool is_simd_optimized = true;

    //=============================================================================
    // 构造函数
    //=============================================================================

    simd_vector() : neon_data(vdupq_n_u16(0)) {}

    explicit simd_vector(uint16_t scalar) : neon_data(vdupq_n_u16(scalar)) {}

    simd_vector(std::initializer_list<uint16_t> init) {
        assert(init.size() == 8);
        auto it = init.begin();
        for (size_t i = 0; i < 8; ++i) {
            data_[i] = *it++;
        }
        neon_data = vld1q_u16(data_);
    }

    simd_vector(const uint16_t* ptr) : neon_data(vld1q_u16(ptr)) {}

    simd_vector(uint16x8_t neon_vec) : neon_data(neon_vec) {}

    static simd_vector load_aligned(const uint16_t* ptr) {
        return simd_vector(vld1q_u16(ptr));
    }

    //=============================================================================
    // 数据访问
    //=============================================================================

    uint16_t& operator[](size_t i) {
        assert(i < 8);
        vst1q_u16(data_, neon_data);
        return data_[i];
    }

    const uint16_t& operator[](size_t i) const {
        assert(i < 8);
        uint16_t* mutable_data = const_cast<uint16_t*>(data_);
        vst1q_u16(mutable_data, neon_data);
        return data_[i];
    }

    uint16_t* data() {
        vst1q_u16(data_, neon_data);
        return data_;
    }

    const uint16_t* data() const {
        uint16_t* mutable_data = const_cast<uint16_t*>(data_);
        vst1q_u16(mutable_data, neon_data);
        return data_;
    }

    constexpr size_t size() const { return 8; }

    void store(uint16_t* ptr) const {
        vst1q_u16(ptr, neon_data);
    }

    void store_aligned(uint16_t* ptr) const {
        vst1q_u16(ptr, neon_data);
    }

    //=============================================================================
    // 算术运算符
    //=============================================================================

    simd_vector& operator+=(const simd_vector& other) {
        neon_data = vaddq_u16(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator-=(const simd_vector& other) {
        neon_data = vsubq_u16(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator*=(const simd_vector& other) {
        neon_data = vmulq_u16(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator+=(uint16_t scalar) {
        neon_data = vaddq_u16(neon_data, vdupq_n_u16(scalar));
        return *this;
    }

    simd_vector& operator-=(uint16_t scalar) {
        neon_data = vsubq_u16(neon_data, vdupq_n_u16(scalar));
        return *this;
    }

    simd_vector& operator*=(uint16_t scalar) {
        neon_data = vmulq_u16(neon_data, vdupq_n_u16(scalar));
        return *this;
    }

    //=============================================================================
    // 比较运算符
    //=============================================================================

    bool operator==(const simd_vector& other) const {
        uint16x8_t result = vceqq_u16(neon_data, other.neon_data);
        uint64x2_t result64 = vpaddlq_u32(vpaddlq_u16(result));
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    bool operator!=(const simd_vector& other) const {
        return !(*this == other);
    }

    // 访问内部NEON数据
    uint16x8_t neon() const { return neon_data; }
};

//=============================================================================
// fp16x8 特化版本 (vec8h) - 仅在支持FP16时启用
//=============================================================================

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
class simd_vector<fp16_t, 8> {
private:
    union {
        float16x8_t neon_data;
        alignas(16) fp16_t data_[8];
    };

public:
    using value_type = fp16_t;
    using size_type = size_t;
    static constexpr size_type size_value = 8;
    static constexpr bool is_simd_optimized = true;

    //=============================================================================
    // 构造函数
    //=============================================================================

    simd_vector() : neon_data(vdupq_n_f16(0.0f)) {}

    explicit simd_vector(fp16_t scalar) : neon_data(vdupq_n_f16(scalar)) {}

    simd_vector(std::initializer_list<fp16_t> init) {
        assert(init.size() == 8);
        auto it = init.begin();
        for (size_t i = 0; i < 8; ++i) {
            data_[i] = *it++;
        }
        neon_data = vld1q_f16(data_);
    }

    simd_vector(const fp16_t* ptr) : neon_data(vld1q_f16(ptr)) {}

    simd_vector(float16x8_t neon_vec) : neon_data(neon_vec) {}

    static simd_vector load_aligned(const fp16_t* ptr) {
        return simd_vector(vld1q_f16(ptr));
    }

    //=============================================================================
    // 数据访问
    //=============================================================================

    fp16_t& operator[](size_t i) {
        assert(i < 8);
        vst1q_f16(data_, neon_data);
        return data_[i];
    }

    const fp16_t& operator[](size_t i) const {
        assert(i < 8);
        fp16_t* mutable_data = const_cast<fp16_t*>(data_);
        vst1q_f16(mutable_data, neon_data);
        return data_[i];
    }

    fp16_t* data() {
        vst1q_f16(data_, neon_data);
        return data_;
    }

    const fp16_t* data() const {
        fp16_t* mutable_data = const_cast<fp16_t*>(data_);
        vst1q_f16(mutable_data, neon_data);
        return data_;
    }

    constexpr size_t size() const { return 8; }

    void store(fp16_t* ptr) const {
        vst1q_f16(ptr, neon_data);
    }

    void store_aligned(fp16_t* ptr) const {
        vst1q_f16(ptr, neon_data);
    }

    //=============================================================================
    // 算术运算符
    //=============================================================================

    simd_vector& operator+=(const simd_vector& other) {
        neon_data = vaddq_f16(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator-=(const simd_vector& other) {
        neon_data = vsubq_f16(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator*=(const simd_vector& other) {
        neon_data = vmulq_f16(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator/=(const simd_vector& other) {
        neon_data = vdivq_f16(neon_data, other.neon_data);
        return *this;
    }

    simd_vector& operator+=(fp16_t scalar) {
        neon_data = vaddq_f16(neon_data, vdupq_n_f16(scalar));
        return *this;
    }

    simd_vector& operator-=(fp16_t scalar) {
        neon_data = vsubq_f16(neon_data, vdupq_n_f16(scalar));
        return *this;
    }

    simd_vector& operator*=(fp16_t scalar) {
        neon_data = vmulq_f16(neon_data, vdupq_n_f16(scalar));
        return *this;
    }

    simd_vector& operator/=(fp16_t scalar) {
        neon_data = vdivq_f16(neon_data, vdupq_n_f16(scalar));
        return *this;
    }

    simd_vector operator-() const {
        return simd_vector(vnegq_f16(neon_data));
    }

    //=============================================================================
    // 比较运算符
    //=============================================================================

    bool operator==(const simd_vector& other) const {
        uint16x8_t result = vceqq_f16(neon_data, other.neon_data);
        uint64x2_t result64 = vpaddlq_u32(vpaddlq_u16(result));
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    bool operator!=(const simd_vector& other) const {
        return !(*this == other);
    }

    // 访问内部NEON数据
    float16x8_t neon() const { return neon_data; }
};
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

//=============================================================================
// NEON特化的算法函数
//=============================================================================

// 点积特化 - float32x2
template<>
inline float dot<float, 2>(const simd_vector<float, 2>& a, const simd_vector<float, 2>& b) {
    float32x2_t mul = vmul_f32(a.neon(), b.neon());
    float32x2_t sum = vpadd_f32(mul, mul);  // 水平相加
    return vget_lane_f32(sum, 0);
}

// 最小值特化 - float32x2
template<>
inline simd_vector<float, 2> min<float, 2>(const simd_vector<float, 2>& a, const simd_vector<float, 2>& b) {
    return simd_vector<float, 2>(vmin_f32(a.neon(), b.neon()));
}

// 最大值特化 - float32x2
template<>
inline simd_vector<float, 2> max<float, 2>(const simd_vector<float, 2>& a, const simd_vector<float, 2>& b) {
    return simd_vector<float, 2>(vmax_f32(a.neon(), b.neon()));
}

// 绝对值特化 - float32x2
template<>
inline simd_vector<float, 2> abs<float, 2>(const simd_vector<float, 2>& v) {
    return simd_vector<float, 2>(vabs_f32(v.neon()));
}

// 点积特化 - float32x4
template<>
inline float dot<float, 4>(const simd_vector<float, 4>& a, const simd_vector<float, 4>& b) {
    float32x4_t mul = vmulq_f32(a.neon(), b.neon());
    float32x2_t sum = vadd_f32(vget_high_f32(mul), vget_low_f32(mul));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}

// 最小值特化 - float32x4
template<>
inline simd_vector<float, 4> min<float, 4>(const simd_vector<float, 4>& a, const simd_vector<float, 4>& b) {
    return simd_vector<float, 4>(vminq_f32(a.neon(), b.neon()));
}

// 最大值特化 - float32x4
template<>
inline simd_vector<float, 4> max<float, 4>(const simd_vector<float, 4>& a, const simd_vector<float, 4>& b) {
    return simd_vector<float, 4>(vmaxq_f32(a.neon(), b.neon()));
}

// 绝对值特化 - float32x4
template<>
inline simd_vector<float, 4> abs<float, 4>(const simd_vector<float, 4>& v) {
    return simd_vector<float, 4>(vabsq_f32(v.neon()));
}

// 最小值特化 - int32x4
template<>
inline simd_vector<int32_t, 4> min<int32_t, 4>(const simd_vector<int32_t, 4>& a, const simd_vector<int32_t, 4>& b) {
    return simd_vector<int32_t, 4>(vminq_s32(a.neon(), b.neon()));
}

// 最大值特化 - int32x4
template<>
inline simd_vector<int32_t, 4> max<int32_t, 4>(const simd_vector<int32_t, 4>& a, const simd_vector<int32_t, 4>& b) {
    return simd_vector<int32_t, 4>(vmaxq_s32(a.neon(), b.neon()));
}

// 绝对值特化 - int32x4
template<>
inline simd_vector<int32_t, 4> abs<int32_t, 4>(const simd_vector<int32_t, 4>& v) {
    return simd_vector<int32_t, 4>(vabsq_s32(v.neon()));
}

// 最小值特化 - uint32x4
template<>
inline simd_vector<uint32_t, 4> min<uint32_t, 4>(const simd_vector<uint32_t, 4>& a, const simd_vector<uint32_t, 4>& b) {
    return simd_vector<uint32_t, 4>(vminq_u32(a.neon(), b.neon()));
}

// 最大值特化 - uint32x4
template<>
inline simd_vector<uint32_t, 4> max<uint32_t, 4>(const simd_vector<uint32_t, 4>& a, const simd_vector<uint32_t, 4>& b) {
    return simd_vector<uint32_t, 4>(vmaxq_u32(a.neon(), b.neon()));
}

// 最小值特化 - uint8x16
template<>
inline simd_vector<uint8_t, 16> min<uint8_t, 16>(const simd_vector<uint8_t, 16>& a, const simd_vector<uint8_t, 16>& b) {
    return simd_vector<uint8_t, 16>(vminq_u8(a.neon(), b.neon()));
}

// 最大值特化 - uint8x16
template<>
inline simd_vector<uint8_t, 16> max<uint8_t, 16>(const simd_vector<uint8_t, 16>& a, const simd_vector<uint8_t, 16>& b) {
    return simd_vector<uint8_t, 16>(vmaxq_u8(a.neon(), b.neon()));
}

// 最小值特化 - uint16x8
template<>
inline simd_vector<uint16_t, 8> min<uint16_t, 8>(const simd_vector<uint16_t, 8>& a, const simd_vector<uint16_t, 8>& b) {
    return simd_vector<uint16_t, 8>(vminq_u16(a.neon(), b.neon()));
}

// 最大值特化 - uint16x8
template<>
inline simd_vector<uint16_t, 8> max<uint16_t, 8>(const simd_vector<uint16_t, 8>& a, const simd_vector<uint16_t, 8>& b) {
    return simd_vector<uint16_t, 8>(vmaxq_u16(a.neon(), b.neon()));
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
// 点积特化 - fp16x8
template<>
inline fp16_t dot<fp16_t, 8>(const simd_vector<fp16_t, 8>& a, const simd_vector<fp16_t, 8>& b) {
    float16x8_t mul = vmulq_f16(a.neon(), b.neon());
    float16x4_t sum = vadd_f16(vget_high_f16(mul), vget_low_f16(mul));
    sum = vpadd_f16(sum, sum);
    sum = vpadd_f16(sum, sum);
    return vget_lane_f16(sum, 0);
}

// 最小值特化 - fp16x8
template<>
inline simd_vector<fp16_t, 8> min<fp16_t, 8>(const simd_vector<fp16_t, 8>& a, const simd_vector<fp16_t, 8>& b) {
    return simd_vector<fp16_t, 8>(vminq_f16(a.neon(), b.neon()));
}

// 最大值特化 - fp16x8
template<>
inline simd_vector<fp16_t, 8> max<fp16_t, 8>(const simd_vector<fp16_t, 8>& a, const simd_vector<fp16_t, 8>& b) {
    return simd_vector<fp16_t, 8>(vmaxq_f16(a.neon(), b.neon()));
}

// 绝对值特化 - fp16x8
template<>
inline simd_vector<fp16_t, 8> abs<fp16_t, 8>(const simd_vector<fp16_t, 8>& v) {
    return simd_vector<fp16_t, 8>(vabsq_f16(v.neon()));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

//=============================================================================
// NEON特化的加宽和饱和运算
//=============================================================================

// uint8x16 加宽加法特化 - 返回寄存器对
template<>
inline simd_vector<uint16_t, 16> add_wide<16>(const simd_vector<uint8_t, 16>& a, const simd_vector<uint8_t, 16>& b) {
    uint8x16_t neon_a = a.neon();
    uint8x16_t neon_b = b.neon();

    // 拆分为低8字节和高8字节，分别进行加宽加法
    uint16x8_t result_low = vaddl_u8(vget_low_u8(neon_a), vget_low_u8(neon_b));
    uint16x8_t result_high = vaddl_u8(vget_high_u8(neon_a), vget_high_u8(neon_b));

    // 构造结果 - 需要16个uint16元素的向量
    simd_vector<uint16_t, 16> result;
    vst1q_u16(result.data(), result_low);        // 存储低8个元素
    vst1q_u16(result.data() + 8, result_high);   // 存储高8个元素
    return result;
}

// uint8x16 加宽乘法特化 - 返回寄存器对
template<>
inline simd_vector<uint16_t, 16> mul_wide<16>(const simd_vector<uint8_t, 16>& a, const simd_vector<uint8_t, 16>& b) {
    uint8x16_t neon_a = a.neon();
    uint8x16_t neon_b = b.neon();

    // 拆分为低8字节和高8字节，分别进行加宽乘法
    uint16x8_t result_low = vmull_u8(vget_low_u8(neon_a), vget_low_u8(neon_b));
    uint16x8_t result_high = vmull_u8(vget_high_u8(neon_a), vget_high_u8(neon_b));

    // 构造结果 - 需要16个uint16元素的向量
    simd_vector<uint16_t, 16> result;
    vst1q_u16(result.data(), result_low);        // 存储低8个元素
    vst1q_u16(result.data() + 8, result_high);   // 存储高8个元素
    return result;
}

// uint16x8 加宽加法特化 - 返回寄存器对
template<>
inline simd_vector<uint32_t, 8> add_wide<8>(const simd_vector<uint16_t, 8>& a, const simd_vector<uint16_t, 8>& b) {
    uint16x8_t neon_a = a.neon();
    uint16x8_t neon_b = b.neon();

    // 拆分为低4个和高4个uint16，分别进行加宽加法
    uint32x4_t result_low = vaddl_u16(vget_low_u16(neon_a), vget_low_u16(neon_b));
    uint32x4_t result_high = vaddl_u16(vget_high_u16(neon_a), vget_high_u16(neon_b));

    // 构造结果 - 需要8个uint32元素的向量
    simd_vector<uint32_t, 8> result;
    vst1q_u32(result.data(), result_low);        // 存储低4个元素
    vst1q_u32(result.data() + 4, result_high);   // 存储高4个元素
    return result;
}

// uint16x8 加宽乘法特化 - 返回寄存器对
template<>
inline simd_vector<uint32_t, 8> mul_wide<8>(const simd_vector<uint16_t, 8>& a, const simd_vector<uint16_t, 8>& b) {
    uint16x8_t neon_a = a.neon();
    uint16x8_t neon_b = b.neon();

    // 拆分为低4个和高4个uint16，分别进行加宽乘法
    uint32x4_t result_low = vmull_u16(vget_low_u16(neon_a), vget_low_u16(neon_b));
    uint32x4_t result_high = vmull_u16(vget_high_u16(neon_a), vget_high_u16(neon_b));

    // 构造结果 - 需要8个uint32元素的向量
    simd_vector<uint32_t, 8> result;
    vst1q_u32(result.data(), result_low);        // 存储低4个元素
    vst1q_u32(result.data() + 4, result_high);   // 存储高4个元素
    return result;
}

// uint8x16 饱和加法特化 - 使用NEON内置饱和指令
template<>
inline simd_vector<uint8_t, 16> add_sat<16>(const simd_vector<uint8_t, 16>& a, const simd_vector<uint8_t, 16>& b) {
    return simd_vector<uint8_t, 16>(vqaddq_u8(a.neon(), b.neon()));
}

// uint8x16 饱和减法特化 - 使用NEON内置饱和指令
template<>
inline simd_vector<uint8_t, 16> sub_sat<16>(const simd_vector<uint8_t, 16>& a, const simd_vector<uint8_t, 16>& b) {
    return simd_vector<uint8_t, 16>(vqsubq_u8(a.neon(), b.neon()));
}

// uint16x8 饱和加法特化 - 使用NEON内置饱和指令
template<>
inline simd_vector<uint16_t, 8> add_sat<8>(const simd_vector<uint16_t, 8>& a, const simd_vector<uint16_t, 8>& b) {
    return simd_vector<uint16_t, 8>(vqaddq_u16(a.neon(), b.neon()));
}

// uint16x8 饱和减法特化 - 使用NEON内置饱和指令
template<>
inline simd_vector<uint16_t, 8> sub_sat<8>(const simd_vector<uint16_t, 8>& a, const simd_vector<uint16_t, 8>& b) {
    return simd_vector<uint16_t, 8>(vqsubq_u16(a.neon(), b.neon()));
}

// 额外添加：从加宽结果饱和转换回原类型的函数
template<>
inline simd_vector<uint8_t, 16> narrow_sat<16>(const simd_vector<uint16_t, 16>& wide_result) {
    // 将16个uint16值饱和转换为16个uint8值
    uint16x8_t low_part = vld1q_u16(wide_result.data());
    uint16x8_t high_part = vld1q_u16(wide_result.data() + 8);

    // 使用饱和窄化指令
    uint8x8_t narrow_low = vqmovn_u16(low_part);
    uint8x8_t narrow_high = vqmovn_u16(high_part);

    // 合并为uint8x16
    uint8x16_t result = vcombine_u8(narrow_low, narrow_high);
    return simd_vector<uint8_t, 16>(result);
}

template<>
inline simd_vector<uint16_t, 8> narrow_sat<8>(const simd_vector<uint32_t, 8>& wide_result) {
    // 将8个uint32值饱和转换为8个uint16值
    uint32x4_t low_part = vld1q_u32(wide_result.data());
    uint32x4_t high_part = vld1q_u32(wide_result.data() + 4);

    // 使用饱和窄化指令
    uint16x4_t narrow_low = vqmovn_u32(low_part);
    uint16x4_t narrow_high = vqmovn_u32(high_part);

    // 合并为uint16x8
    uint16x8_t result = vcombine_u16(narrow_low, narrow_high);
    return simd_vector<uint16_t, 8>(result);
}

} // namespace tiny_simd

#endif // TINY_SIMD_ARM_NEON

#endif // TINY_SIMD_HPP