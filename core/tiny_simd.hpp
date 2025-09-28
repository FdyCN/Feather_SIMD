#ifndef TINY_SIMD_HPP
#define TINY_SIMD_HPP

#include <cstddef>
#include <type_traits>
#include <cmath>
#include <cassert>
#include <array>
#include <initializer_list>

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
static constexpr size_t max_vector_size_int32 = 4;
static constexpr size_t max_vector_size_int16 = 8;
static constexpr size_t max_vector_size_int8 = 16;
#elif defined(TINY_SIMD_X86_AVX2)
static constexpr size_t max_vector_size_float = 8;
static constexpr size_t max_vector_size_double = 4;
static constexpr size_t max_vector_size_int32 = 8;
static constexpr size_t max_vector_size_int16 = 16;
static constexpr size_t max_vector_size_int8 = 32;
#elif defined(TINY_SIMD_X86_AVX)
static constexpr size_t max_vector_size_float = 8;
static constexpr size_t max_vector_size_double = 4;
static constexpr size_t max_vector_size_int32 = 4;
static constexpr size_t max_vector_size_int16 = 8;
static constexpr size_t max_vector_size_int8 = 16;
#elif defined(TINY_SIMD_X86_SSE)
static constexpr size_t max_vector_size_float = 4;
static constexpr size_t max_vector_size_double = 2;
static constexpr size_t max_vector_size_int32 = 4;
static constexpr size_t max_vector_size_int16 = 8;
static constexpr size_t max_vector_size_int8 = 16;
#else
static constexpr size_t max_vector_size_float = 1;
static constexpr size_t max_vector_size_double = 1;
static constexpr size_t max_vector_size_int32 = 1;
static constexpr size_t max_vector_size_int16 = 1;
static constexpr size_t max_vector_size_int8 = 1;
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
    static constexpr bool is_simd_optimized = (N == 4 && std::is_same<T, float>::value) ||
                                             (N == 4 && std::is_same<T, int32_t>::value) ||
                                             (N == 8 && std::is_same<T, int16_t>::value) ||
                                             (N == 16 && std::is_same<T, int8_t>::value);
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

using vec4i = simd_vector<int32_t, 4>;
using vec8i = simd_vector<int32_t, 8>;

using vec8s = simd_vector<int16_t, 8>;
using vec16s = simd_vector<int16_t, 16>;

using vec16b = simd_vector<int8_t, 16>;
using vec32b = simd_vector<int8_t, 32>;

//=============================================================================
// 基础算法函数
//=============================================================================

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
// NEON特化的算法函数
//=============================================================================

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

} // namespace tiny_simd

#endif // TINY_SIMD_ARM_NEON

#endif // TINY_SIMD_HPP