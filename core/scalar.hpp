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

    //=========================================================================
    // Vector Splitting Operations (get_low / get_high)
    //=========================================================================

    // Extract low half - returns half-size register
    static scalar_register<T, N/2> get_low(reg_type a) {
        static_assert(N % 2 == 0, "Vector size must be even for get_low");
        scalar_register<T, N/2> result;
        for (size_t i = 0; i < N/2; ++i) {
            result.data[i] = a.data[i];
        }
        return result;
    }

    // Extract high half - returns half-size register
    static scalar_register<T, N/2> get_high(reg_type a) {
        static_assert(N % 2 == 0, "Vector size must be even for get_high");
        scalar_register<T, N/2> result;
        for (size_t i = 0; i < N/2; ++i) {
            result.data[i] = a.data[N/2 + i];
        }
        return result;
    }

    //=========================================================================
    // Type Conversion Operations
    //=========================================================================

    // Integer -> Float conversion (only for integer types)
    template<typename U = T>
    static typename std::enable_if<std::is_integral<U>::value, scalar_register<float, N>>::type
    convert_to_float(reg_type a) {
        scalar_register<float, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = static_cast<float>(a.data[i]);
        }
        return result;
    }

    // Float -> Integer conversion (with rounding, only for float type)
    template<typename IntT, typename U = T>
    static typename std::enable_if<std::is_same<U, float>::value, scalar_register<IntT, N>>::type
    convert_to_int(reg_type a) {
        static_assert(std::is_integral<IntT>::value, "Target type must be integer");
        scalar_register<IntT, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = static_cast<IntT>(std::round(a.data[i]));
        }
        return result;
    }

    // fp16 -> fp32 conversion (only for fp16_t type)
    template<typename U = T>
    static typename std::enable_if<std::is_same<U, fp16_t>::value, scalar_register<float, N>>::type
    convert_to_fp32(reg_type a) {
        scalar_register<float, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = static_cast<float>(a.data[i]);
        }
        return result;
    }

    // fp32 -> fp16 conversion (only for float type)
    template<typename U = T>
    static typename std::enable_if<std::is_same<U, float>::value, scalar_register<fp16_t, N>>::type
    convert_to_fp16(reg_type a) {
        scalar_register<fp16_t, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = static_cast<fp16_t>(a.data[i]);
        }
        return result;
    }

    //=========================================================================
    // Integer Width Conversions (Phase 2)
    //=========================================================================

    // Widening: integer types to twice their size
    // Works for: int8→int16, uint8→uint16, int16→int32, uint16→uint32, int32→int64, uint32→uint64
    template<typename U = T>
    static typename std::enable_if<
        std::is_integral<U>::value && (sizeof(U) < 8),
        scalar_register<typename std::conditional<std::is_signed<U>::value,
            typename std::make_signed<typename std::conditional<sizeof(U)==1, int16_t,
                typename std::conditional<sizeof(U)==2, int32_t, int64_t>::type>::type>::type,
            typename std::make_unsigned<typename std::conditional<sizeof(U)==1, uint16_t,
                typename std::conditional<sizeof(U)==2, uint32_t, uint64_t>::type>::type>::type
        >::type, N>
    >::type
    convert_widen(reg_type a) {
        using target_type = typename std::conditional<std::is_signed<U>::value,
            typename std::make_signed<typename std::conditional<sizeof(U)==1, int16_t,
                typename std::conditional<sizeof(U)==2, int32_t, int64_t>::type>::type>::type,
            typename std::make_unsigned<typename std::conditional<sizeof(U)==1, uint16_t,
                typename std::conditional<sizeof(U)==2, uint32_t, uint64_t>::type>::type>::type
        >::type;

        scalar_register<target_type, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = static_cast<target_type>(a.data[i]);
        }
        return result;
    }

    // Narrowing: integer types to half their size (may overflow)
    // Works for: int16→int8, uint16→uint8, int32→int16, uint32→uint16, int64→int32, uint64→uint32
    template<typename U = T>
    static typename std::enable_if<
        std::is_integral<U>::value && (sizeof(U) > 1),
        scalar_register<typename std::conditional<std::is_signed<U>::value,
            typename std::make_signed<typename std::conditional<sizeof(U)==2, int8_t,
                typename std::conditional<sizeof(U)==4, int16_t, int32_t>::type>::type>::type,
            typename std::make_unsigned<typename std::conditional<sizeof(U)==2, uint8_t,
                typename std::conditional<sizeof(U)==4, uint16_t, uint32_t>::type>::type>::type
        >::type, N>
    >::type
    convert_narrow(reg_type a) {
        using target_type = typename std::conditional<std::is_signed<U>::value,
            typename std::make_signed<typename std::conditional<sizeof(U)==2, int8_t,
                typename std::conditional<sizeof(U)==4, int16_t, int32_t>::type>::type>::type,
            typename std::make_unsigned<typename std::conditional<sizeof(U)==2, uint8_t,
                typename std::conditional<sizeof(U)==4, uint16_t, uint32_t>::type>::type>::type
        >::type;

        scalar_register<target_type, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = static_cast<target_type>(a.data[i]);
        }
        return result;
    }

    // Narrowing with saturation: prevent overflow
    template<typename U = T>
    static typename std::enable_if<
        std::is_integral<U>::value && (sizeof(U) > 1),
        scalar_register<typename std::conditional<std::is_signed<U>::value,
            typename std::make_signed<typename std::conditional<sizeof(U)==2, int8_t,
                typename std::conditional<sizeof(U)==4, int16_t, int32_t>::type>::type>::type,
            typename std::make_unsigned<typename std::conditional<sizeof(U)==2, uint8_t,
                typename std::conditional<sizeof(U)==4, uint16_t, uint32_t>::type>::type>::type
        >::type, N>
    >::type
    convert_narrow_sat(reg_type a) {
        using target_type = typename std::conditional<std::is_signed<U>::value,
            typename std::make_signed<typename std::conditional<sizeof(U)==2, int8_t,
                typename std::conditional<sizeof(U)==4, int16_t, int32_t>::type>::type>::type,
            typename std::make_unsigned<typename std::conditional<sizeof(U)==2, uint8_t,
                typename std::conditional<sizeof(U)==4, uint16_t, uint32_t>::type>::type>::type
        >::type;

        constexpr target_type min_val = std::numeric_limits<target_type>::min();
        constexpr target_type max_val = std::numeric_limits<target_type>::max();

        scalar_register<target_type, N> result;
        for (size_t i = 0; i < N; ++i) {
            U val = a.data[i];
            if (val < min_val) {
                result.data[i] = min_val;
            } else if (val > max_val) {
                result.data[i] = max_val;
            } else {
                result.data[i] = static_cast<target_type>(val);
            }
        }
        return result;
    }

    //=========================================================================
    // Phase 3: Unsigned to Signed Conversions
    //=========================================================================

    // Same-width unsigned to signed conversion
    template<typename U = T>
    static typename std::enable_if<
        std::is_unsigned<U>::value,
        scalar_register<typename std::make_signed<U>::type, N>
    >::type
    convert_to_signed(reg_type a) {
        using target_type = typename std::make_signed<U>::type;
        scalar_register<target_type, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.data[i] = static_cast<target_type>(a.data[i]);
        }
        return result;
    }

    // Saturating narrowing to signed
    template<typename U = T>
    static typename std::enable_if<
        std::is_unsigned<U>::value && (sizeof(U) > 1),
        scalar_register<typename std::make_signed<typename std::conditional<sizeof(U)==2, uint8_t,
            typename std::conditional<sizeof(U)==4, uint16_t, uint32_t>::type>::type>::type, N>
    >::type
    convert_to_signed_sat(reg_type a) {
        // Determine unsigned narrow type
        using unsigned_narrow = typename std::conditional<sizeof(U)==2, uint8_t,
            typename std::conditional<sizeof(U)==4, uint16_t, uint32_t>::type>::type;
        // Then make it signed
        using target_type = typename std::make_signed<unsigned_narrow>::type;

        constexpr target_type max_val = std::numeric_limits<target_type>::max();

        scalar_register<target_type, N> result;
        for (size_t i = 0; i < N; ++i) {
            U val = a.data[i];
            // Saturate to [0, max_signed]
            if (val > static_cast<U>(max_val)) {
                result.data[i] = max_val;
            } else {
                result.data[i] = static_cast<target_type>(val);
            }
        }
        return result;
    }
};

} // namespace tiny_simd

#endif // TINY_SIMD_SCALAR_HPP
