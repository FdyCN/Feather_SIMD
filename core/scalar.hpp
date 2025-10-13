#ifndef TINY_SIMD_SCALAR_HPP
#define TINY_SIMD_SCALAR_HPP

#include "base.hpp"
#include <algorithm>
#include <cmath>

namespace tiny_simd {

//=============================================================================
// Scalar Register Type - Simple array wrapper
//=============================================================================

template<typename T, size_t N>
struct scalar_register {
    alignas(config::simd_alignment) T data[N];
};

//=============================================================================
// Scalar Backend Operations Implementation
//=============================================================================

template<typename T, size_t N>
struct backend_ops<scalar_backend, T, N> {
    using reg_type = scalar_register<T, N>;

    //=========================================================================
    // Load/Store Operations
    //=========================================================================

    static reg_type zero() {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = T{0};
        }
        return result;
    }

    static reg_type set1(T scalar) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = scalar;
        }
        return result;
    }

    static reg_type load(const T* ptr) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = ptr[i];
        }
        return result;
    }

    static reg_type load_aligned(const T* ptr) {
        return load(ptr);  // Same as unaligned for scalar
    }

    static reg_type load_from_initializer(std::initializer_list<T> init) {
        reg_type result;
        auto it = init.begin();
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = (it != init.end()) ? *it++ : T{0};
        }
        return result;
    }

    static void store(T* ptr, reg_type reg) {
        for (size_t i = 0; i < N; ++i) {
            ptr[i] = reg.data[i];
        }
    }

    static void store_aligned(T* ptr, reg_type reg) {
        store(ptr, reg);  // Same as unaligned for scalar
    }

    static T extract(reg_type reg, size_t index) {
        return reg.data[index];
    }

    //=========================================================================
    // Arithmetic Operations
    //=========================================================================

    static reg_type add(reg_type a, reg_type b) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = a.data[i] + b.data[i];
        }
        return result;
    }

    static reg_type sub(reg_type a, reg_type b) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = a.data[i] - b.data[i];
        }
        return result;
    }

    static reg_type mul(reg_type a, reg_type b) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = a.data[i] * b.data[i];
        }
        return result;
    }

    static reg_type div(reg_type a, reg_type b) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = a.data[i] / b.data[i];
        }
        return result;
    }

    static reg_type neg(reg_type a) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = -a.data[i];
        }
        return result;
    }

    //=========================================================================
    // Comparison Operations
    //=========================================================================

    static bool equal(reg_type a, reg_type b) {
        for (size_t i = 0; i < N; ++i) {
            if (a.data[i] != b.data[i]) return false;
        }
        return true;
    }

    //=========================================================================
    // Min/Max Operations
    //=========================================================================

    static reg_type min(reg_type a, reg_type b) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = std::min(a.data[i], b.data[i]);
        }
        return result;
    }

    static reg_type max(reg_type a, reg_type b) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = std::max(a.data[i], b.data[i]);
        }
        return result;
    }

    static reg_type abs(reg_type a) {
        reg_type result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = std::abs(a.data[i]);
        }
        return result;
    }
};

} // namespace tiny_simd

#endif // TINY_SIMD_SCALAR_HPP
