/*
 * Copyright (c) 2025 FdyCN
 *
 * Distributed under MIT license.
 * See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
 */

#ifndef TINY_SIMD_NEON_HPP
#define TINY_SIMD_NEON_HPP

#include "base.hpp"

#ifdef TINY_SIMD_ARM_NEON
#include <arm_neon.h>

namespace tiny_simd {

//=============================================================================
// NEON Register Type Traits
//=============================================================================

// Helper: Map (T, N) to NEON register type
template<typename T, size_t N> struct neon_traits;

// float specializations
template<> struct neon_traits<float, 2> { using reg_type = float32x2_t; };
template<> struct neon_traits<float, 4> { using reg_type = float32x4_t; };

// int32 specializations
template<> struct neon_traits<int32_t, 4> { using reg_type = int32x4_t; };

// uint32 specializations
template<> struct neon_traits<uint32_t, 4> { using reg_type = uint32x4_t; };

// int16 specializations
template<> struct neon_traits<int16_t, 8> { using reg_type = int16x8_t; };

// uint16 specializations
template<> struct neon_traits<uint16_t, 8> { using reg_type = uint16x8_t; };

// int8 specializations
template<> struct neon_traits<int8_t, 16> { using reg_type = int8x16_t; };

// uint8 specializations
template<> struct neon_traits<uint8_t, 16> { using reg_type = uint8x16_t; };

// fp16 specializations (if supported)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<> struct neon_traits<fp16_t, 8> { using reg_type = float16x8_t; };
#endif

//=============================================================================
// NEON Backend Operations - float32x2
//=============================================================================

template<>
struct backend_ops<neon_backend, float, 2> {
    using reg_type = float32x2_t;

    static reg_type zero() { return vdup_n_f32(0.0f); }
    static reg_type set1(float scalar) { return vdup_n_f32(scalar); }
    static reg_type load(const float* ptr) { return vld1_f32(ptr); }
    static reg_type load_aligned(const float* ptr) { return vld1_f32(ptr); }

    static reg_type load_from_initializer(std::initializer_list<float> init) {
        alignas(8) float temp[2] = {0.0f, 0.0f};
        auto it = init.begin();
        for (size_t i = 0; i < 2 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1_f32(temp);
    }

    static void store(float* ptr, reg_type reg) { vst1_f32(ptr, reg); }
    static void store_aligned(float* ptr, reg_type reg) { vst1_f32(ptr, reg); }

    static float extract(reg_type reg, size_t index) {
        alignas(8) float temp[2];
        vst1_f32(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vadd_f32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsub_f32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmul_f32(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        // Use reciprocal approximation + Newton-Raphson refinement
        float32x2_t reciprocal = vrecpe_f32(b);
        reciprocal = vmul_f32(vrecps_f32(b, reciprocal), reciprocal);
        return vmul_f32(a, reciprocal);
    }

    static reg_type neg(reg_type a) { return vneg_f32(a); }

    static bool equal(reg_type a, reg_type b) {
        uint32x2_t result = vceq_f32(a, b);
        return vget_lane_u32(result, 0) == 0xFFFFFFFF &&
               vget_lane_u32(result, 1) == 0xFFFFFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return vmin_f32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmax_f32(a, b); }
    static reg_type abs(reg_type a) { return vabs_f32(a); }

    static reg_type fma(reg_type a, reg_type b, reg_type c) {
        // vfma_f32(acc, a, b) -> acc + a * b
        // We want a * b + c, so we pass c as accumulator
        return vfma_f32(c, a, b);
    }
};

//=============================================================================
// NEON Backend Operations - float32x4
//=============================================================================

template<>
struct backend_ops<neon_backend, float, 4> {
    using reg_type = float32x4_t;

    static reg_type zero() { return vdupq_n_f32(0.0f); }
    static reg_type set1(float scalar) { return vdupq_n_f32(scalar); }
    static reg_type load(const float* ptr) { return vld1q_f32(ptr); }
    static reg_type load_aligned(const float* ptr) { return vld1q_f32(ptr); }

    static reg_type load_from_initializer(std::initializer_list<float> init) {
        alignas(16) float temp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1q_f32(temp);
    }

    static void store(float* ptr, reg_type reg) { vst1q_f32(ptr, reg); }
    static void store_aligned(float* ptr, reg_type reg) { vst1q_f32(ptr, reg); }

    static float extract(reg_type reg, size_t index) {
        alignas(16) float temp[4];
        vst1q_f32(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vaddq_f32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsubq_f32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmulq_f32(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        float32x4_t reciprocal = vrecpeq_f32(b);
        reciprocal = vmulq_f32(vrecpsq_f32(b, reciprocal), reciprocal);
        return vmulq_f32(a, reciprocal);
    }

    static reg_type neg(reg_type a) { return vnegq_f32(a); }

    static bool equal(reg_type a, reg_type b) {
        uint32x4_t result = vceqq_f32(a, b);
        uint64x2_t result64 = vpaddlq_u32(result);
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vminq_f32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmaxq_f32(a, b); }
    static reg_type abs(reg_type a) { return vabsq_f32(a); }

    static reg_type fma(reg_type a, reg_type b, reg_type c) {
        // vfmaq_f32(acc, a, b) -> acc + a * b
        return vfmaq_f32(c, a, b);
    }

    // Vector splitting operations
    static float32x2_t get_low(reg_type a) { return vget_low_f32(a); }
    static float32x2_t get_high(reg_type a) { return vget_high_f32(a); }

    //-------------------------------------------------------------------------
    // Type Conversion Operations
    //-------------------------------------------------------------------------

    // float32 -> int32 (round to nearest)
    template<typename TargetT>
    static typename std::enable_if<std::is_same<TargetT, int32_t>::value, int32x4_t>::type
    convert_to_int(reg_type a) {
        return vcvtq_s32_f32(a);  // Single NEON instruction
    }

    // float32 -> uint32 (round to nearest)
    template<typename TargetT>
    static typename std::enable_if<std::is_same<TargetT, uint32_t>::value, uint32x4_t>::type
    convert_to_int(reg_type a) {
        return vcvtq_u32_f32(a);  // Single NEON instruction
    }

    // float32 -> fp16 (4个fp32 -> 低4个fp16)
    static float16x4_t convert_to_fp16(reg_type a) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        return vcvt_f16_f32(a);  // Single instruction: float32x4 -> float16x4
#else
        // Fallback: use scalar conversion for platforms without native fp16
        alignas(16) float temp[4];
        vst1q_f32(temp, a);
        float16x4_t result;
        fp16_t* result_ptr = reinterpret_cast<fp16_t*>(&result);
        for (int i = 0; i < 4; ++i) {
            result_ptr[i] = static_cast<fp16_t>(temp[i]);
        }
        return result;
#endif
    }
};

//=============================================================================
// NEON Backend Operations - int32x4
//=============================================================================

template<>
struct backend_ops<neon_backend, int32_t, 4> {
    using reg_type = int32x4_t;

    static reg_type zero() { return vdupq_n_s32(0); }
    static reg_type set1(int32_t scalar) { return vdupq_n_s32(scalar); }
    static reg_type load(const int32_t* ptr) { return vld1q_s32(ptr); }
    static reg_type load_aligned(const int32_t* ptr) { return vld1q_s32(ptr); }

    static reg_type load_from_initializer(std::initializer_list<int32_t> init) {
        alignas(16) int32_t temp[4] = {0, 0, 0, 0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1q_s32(temp);
    }

    static void store(int32_t* ptr, reg_type reg) { vst1q_s32(ptr, reg); }
    static void store_aligned(int32_t* ptr, reg_type reg) { vst1q_s32(ptr, reg); }

    static int32_t extract(reg_type reg, size_t index) {
        alignas(16) int32_t temp[4];
        vst1q_s32(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vaddq_s32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsubq_s32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmulq_s32(a, b); }

    // Division: NEON doesn't have int division, fall back to scalar
    static reg_type div(reg_type a, reg_type b) {
        alignas(16) int32_t temp_a[4], temp_b[4], result[4];
        vst1q_s32(temp_a, a);
        vst1q_s32(temp_b, b);
        for (size_t i = 0; i < 4; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1q_s32(result);
    }

    static reg_type neg(reg_type a) { return vnegq_s32(a); }

    static bool equal(reg_type a, reg_type b) {
        uint32x4_t result = vceqq_s32(a, b);
        uint64x2_t result64 = vpaddlq_u32(result);
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vminq_s32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmaxq_s32(a, b); }
    static reg_type abs(reg_type a) { return vabsq_s32(a); }

    // Bitwise Operations
    static reg_type bitwise_and(reg_type a, reg_type b) { return vandq_s32(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return vorrq_s32(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return veorq_s32(a, b); }
    static reg_type bitwise_not(reg_type a) { return vmvnq_s32(a); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return vbicq_s32(a, b); }

    // Shift Operations
    static reg_type shift_left(reg_type a, int count) {
        return vshlq_s32(a, vdupq_n_s32(count));
    }
    static reg_type shift_right(reg_type a, int count) {
        // Negative shift amount for right shift
        return vshlq_s32(a, vdupq_n_s32(-count));
    }

    // Vector splitting operations
    static int32x2_t get_low(reg_type a) { return vget_low_s32(a); }
    static int32x2_t get_high(reg_type a) { return vget_high_s32(a); }

    //-------------------------------------------------------------------------
    // Type Conversion Operations
    //-------------------------------------------------------------------------

    // int32 -> float32
    static float32x4_t convert_to_float(reg_type a) {
        return vcvtq_f32_s32(a);  // Single NEON instruction
    }
};

//=============================================================================
// NEON Backend Operations - uint32x4
//=============================================================================

template<>
struct backend_ops<neon_backend, uint32_t, 4> {
    using reg_type = uint32x4_t;

    static reg_type zero() { return vdupq_n_u32(0); }
    static reg_type set1(uint32_t scalar) { return vdupq_n_u32(scalar); }
    static reg_type load(const uint32_t* ptr) { return vld1q_u32(ptr); }
    static reg_type load_aligned(const uint32_t* ptr) { return vld1q_u32(ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint32_t> init) {
        alignas(16) uint32_t temp[4] = {0, 0, 0, 0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1q_u32(temp);
    }

    static void store(uint32_t* ptr, reg_type reg) { vst1q_u32(ptr, reg); }
    static void store_aligned(uint32_t* ptr, reg_type reg) { vst1q_u32(ptr, reg); }

    static uint32_t extract(reg_type reg, size_t index) {
        alignas(16) uint32_t temp[4];
        vst1q_u32(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vaddq_u32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsubq_u32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmulq_u32(a, b); }

    // Division: fall back to scalar
    static reg_type div(reg_type a, reg_type b) {
        alignas(16) uint32_t temp_a[4], temp_b[4], result[4];
        vst1q_u32(temp_a, a);
        vst1q_u32(temp_b, b);
        for (size_t i = 0; i < 4; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1q_u32(result);
    }

    static reg_type neg(reg_type a) {
        // Negate by subtracting from zero
        return vsubq_u32(vdupq_n_u32(0), a);
    }

    static bool equal(reg_type a, reg_type b) {
        uint32x4_t result = vceqq_u32(a, b);
        uint64x2_t result64 = vpaddlq_u32(result);
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vminq_u32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmaxq_u32(a, b); }

    static reg_type abs(reg_type a) { return a; } // unsigned, already absolute

    // Bitwise Operations
    static reg_type bitwise_and(reg_type a, reg_type b) { return vandq_u32(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return vorrq_u32(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return veorq_u32(a, b); }
    static reg_type bitwise_not(reg_type a) { return vmvnq_u32(a); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return vbicq_u32(a, b); }

    // Shift Operations
    static reg_type shift_left(reg_type a, int count) {
        return vshlq_u32(a, vdupq_n_s32(count));
    }
    static reg_type shift_right(reg_type a, int count) {
        // Negative shift amount for right shift (Logical shift for unsigned)
        return vshlq_u32(a, vdupq_n_s32(-count));
    }

    // Vector splitting operations
    static uint32x2_t get_low(reg_type a) { return vget_low_u32(a); }
    static uint32x2_t get_high(reg_type a) { return vget_high_u32(a); }

    //-------------------------------------------------------------------------
    // Type Conversion Operations
    //-------------------------------------------------------------------------

    // uint32 -> float32
    static float32x4_t convert_to_float(reg_type a) {
        return vcvtq_f32_u32(a);  // Single NEON instruction
    }

    // uint32 -> int32 (same width, reinterpret)
    static int32x4_t convert_to_signed(reg_type a) {
        return vreinterpretq_s32_u32(a);  // 0 cycle (just type alias)
    }
};

//=============================================================================
// NEON Backend Operations - uint16x8
//=============================================================================

template<>
struct backend_ops<neon_backend, uint16_t, 8> {
    using reg_type = uint16x8_t;

    static reg_type zero() { return vdupq_n_u16(0); }
    static reg_type set1(uint16_t scalar) { return vdupq_n_u16(scalar); }
    static reg_type load(const uint16_t* ptr) { return vld1q_u16(ptr); }
    static reg_type load_aligned(const uint16_t* ptr) { return vld1q_u16(ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint16_t> init) {
        alignas(16) uint16_t temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1q_u16(temp);
    }

    static void store(uint16_t* ptr, reg_type reg) { vst1q_u16(ptr, reg); }
    static void store_aligned(uint16_t* ptr, reg_type reg) { vst1q_u16(ptr, reg); }

    static uint16_t extract(reg_type reg, size_t index) {
        alignas(16) uint16_t temp[8];
        vst1q_u16(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vaddq_u16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsubq_u16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmulq_u16(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(16) uint16_t temp_a[8], temp_b[8], result[8];
        vst1q_u16(temp_a, a);
        vst1q_u16(temp_b, b);
        for (size_t i = 0; i < 8; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1q_u16(result);
    }

    static reg_type neg(reg_type a) {
        return vsubq_u16(vdupq_n_u16(0), a);
    }

    static bool equal(reg_type a, reg_type b) {
        uint16x8_t result = vceqq_u16(a, b);
        uint64x2_t result64 = vpaddlq_u32(vpaddlq_u16(result));
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vminq_u16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmaxq_u16(a, b); }
    static reg_type abs(reg_type a) { return a; }

    // Bitwise Operations
    static reg_type bitwise_and(reg_type a, reg_type b) { return vandq_u16(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return vorrq_u16(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return veorq_u16(a, b); }
    static reg_type bitwise_not(reg_type a) { return vmvnq_u16(a); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return vbicq_u16(a, b); }

    // Shift Operations
    static reg_type shift_left(reg_type a, int count) {
        return vshlq_u16(a, vdupq_n_s16(static_cast<int16_t>(count)));
    }
    static reg_type shift_right(reg_type a, int count) {
        return vshlq_u16(a, vdupq_n_s16(static_cast<int16_t>(-count)));
    }

    // Vector splitting operations
    static uint16x4_t get_low(reg_type a) { return vget_low_u16(a); }
    static uint16x4_t get_high(reg_type a) { return vget_high_u16(a); }

    //-------------------------------------------------------------------------
    // Type Conversion Operations
    //-------------------------------------------------------------------------

    // Narrowing: uint16x8 → uint8x8
    static uint8x8_t convert_narrow(reg_type a) {
        return vmovn_u16(a);  // Single NEON instruction (1 cycle)
    }

    // Narrowing with saturation: uint16x8 → uint8x8
    static uint8x8_t convert_narrow_sat(reg_type a) {
        return vqmovn_u16(a);  // Saturate to [0, 255]
    }

    // uint16 -> int16 (same width, reinterpret)
    static int16x8_t convert_to_signed(reg_type a) {
        return vreinterpretq_s16_u16(a);  // 0 cycle (just type alias)
    }

    // uint16 -> int8 (narrowing with saturation to signed range)
    static int8x8_t convert_to_signed_sat(reg_type a) {
        // Strategy: First saturate to [0, 127], then reinterpret as signed
        // vqmovn_u16 saturates to [0, 255], but we need [0, 127]
        // Use vmin to clamp to 127 first
        uint16x8_t clamped = vminq_u16(a, vdupq_n_u16(127));
        uint8x8_t narrow = vmovn_u16(clamped);
        return vreinterpret_s8_u8(narrow);
    }

    // Widening: uint16x4 (from low half) → uint32x4
    // Note: 仅对 64-bit narrow register 有效,这里预留接口
};

//=============================================================================
// NEON Backend Operations - int16x8
//=============================================================================

template<>
struct backend_ops<neon_backend, int16_t, 8> {
    using reg_type = int16x8_t;

    static reg_type zero() { return vdupq_n_s16(0); }
    static reg_type set1(int16_t scalar) { return vdupq_n_s16(scalar); }
    static reg_type load(const int16_t* ptr) { return vld1q_s16(ptr); }
    static reg_type load_aligned(const int16_t* ptr) { return vld1q_s16(ptr); }

    static reg_type load_from_initializer(std::initializer_list<int16_t> init) {
        alignas(16) int16_t temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1q_s16(temp);
    }

    static void store(int16_t* ptr, reg_type reg) { vst1q_s16(ptr, reg); }
    static void store_aligned(int16_t* ptr, reg_type reg) { vst1q_s16(ptr, reg); }

    static int16_t extract(reg_type reg, size_t index) {
        alignas(16) int16_t temp[8];
        vst1q_s16(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vaddq_s16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsubq_s16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmulq_s16(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(16) int16_t temp_a[8], temp_b[8], result[8];
        vst1q_s16(temp_a, a);
        vst1q_s16(temp_b, b);
        for (size_t i = 0; i < 8; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1q_s16(result);
    }

    static reg_type neg(reg_type a) { return vnegq_s16(a); }

    static bool equal(reg_type a, reg_type b) {
        uint16x8_t result = vceqq_s16(a, b);
        uint64x2_t result64 = vpaddlq_u32(vpaddlq_u16(result));
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vminq_s16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmaxq_s16(a, b); }
    static reg_type abs(reg_type a) { return vabsq_s16(a); }

    // Bitwise Operations
    static reg_type bitwise_and(reg_type a, reg_type b) { return vandq_s16(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return vorrq_s16(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return veorq_s16(a, b); }
    static reg_type bitwise_not(reg_type a) { return vmvnq_s16(a); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return vbicq_s16(a, b); }

    // Shift Operations
    static reg_type shift_left(reg_type a, int count) {
        return vshlq_s16(a, vdupq_n_s16(static_cast<int16_t>(count)));
    }
    static reg_type shift_right(reg_type a, int count) {
        return vshlq_s16(a, vdupq_n_s16(static_cast<int16_t>(-count)));
    }

    // Vector splitting operations
    static int16x4_t get_low(reg_type a) { return vget_low_s16(a); }
    static int16x4_t get_high(reg_type a) { return vget_high_s16(a); }

    //-------------------------------------------------------------------------
    // Type Conversion Operations
    //-------------------------------------------------------------------------

    // Narrowing: int16x8 → int8x8
    static int8x8_t convert_narrow(reg_type a) {
        return vmovn_s16(a);  // Single NEON instruction (1 cycle)
    }

    // Narrowing with saturation: int16x8 → int8x8
    static int8x8_t convert_narrow_sat(reg_type a) {
        return vqmovn_s16(a);  // Saturate to [-128, 127]
    }
};

//=============================================================================
// NEON Backend Operations - uint8x16
//=============================================================================

template<>
struct backend_ops<neon_backend, uint8_t, 16> {
    using reg_type = uint8x16_t;

    static reg_type zero() { return vdupq_n_u8(0); }
    static reg_type set1(uint8_t scalar) { return vdupq_n_u8(scalar); }
    static reg_type load(const uint8_t* ptr) { return vld1q_u8(ptr); }
    static reg_type load_aligned(const uint8_t* ptr) { return vld1q_u8(ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint8_t> init) {
        alignas(16) uint8_t temp[16] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 16 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1q_u8(temp);
    }

    static void store(uint8_t* ptr, reg_type reg) { vst1q_u8(ptr, reg); }
    static void store_aligned(uint8_t* ptr, reg_type reg) { vst1q_u8(ptr, reg); }

    static uint8_t extract(reg_type reg, size_t index) {
        alignas(16) uint8_t temp[16];
        vst1q_u8(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vaddq_u8(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsubq_u8(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmulq_u8(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(16) uint8_t temp_a[16], temp_b[16], result[16];
        vst1q_u8(temp_a, a);
        vst1q_u8(temp_b, b);
        for (size_t i = 0; i < 16; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1q_u8(result);
    }

    static reg_type neg(reg_type a) {
        return vsubq_u8(vdupq_n_u8(0), a);
    }

    static bool equal(reg_type a, reg_type b) {
        uint8x16_t result = vceqq_u8(a, b);
        uint16x8_t result16 = vpaddlq_u8(result);
        uint32x4_t result32 = vpaddlq_u16(result16);
        uint64x2_t result64 = vpaddlq_u32(result32);
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vminq_u8(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmaxq_u8(a, b); }
    static reg_type abs(reg_type a) { return a; }

    // Bitwise Operations
    static reg_type bitwise_and(reg_type a, reg_type b) { return vandq_u8(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return vorrq_u8(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return veorq_u8(a, b); }
    static reg_type bitwise_not(reg_type a) { return vmvnq_u8(a); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return vbicq_u8(a, b); }

    // Shift Operations
    static reg_type shift_left(reg_type a, int count) {
        return vshlq_u8(a, vdupq_n_s8(static_cast<int8_t>(count)));
    }
    static reg_type shift_right(reg_type a, int count) {
        return vshlq_u8(a, vdupq_n_s8(static_cast<int8_t>(-count)));
    }

    // Vector splitting operations
    static uint8x8_t get_low(reg_type a) { return vget_low_u8(a); }
    static uint8x8_t get_high(reg_type a) { return vget_high_u8(a); }

    // uint8 -> int8 (same width, reinterpret)
    static int8x16_t convert_to_signed(reg_type a) {
        return vreinterpretq_s8_u8(a);  // 0 cycle (just type alias)
    }
};

//=============================================================================
// NEON Backend Operations - int8x16
//=============================================================================

template<>
struct backend_ops<neon_backend, int8_t, 16> {
    using reg_type = int8x16_t;

    static reg_type zero() { return vdupq_n_s8(0); }
    static reg_type set1(int8_t scalar) { return vdupq_n_s8(scalar); }
    static reg_type load(const int8_t* ptr) { return vld1q_s8(ptr); }
    static reg_type load_aligned(const int8_t* ptr) { return vld1q_s8(ptr); }

    static reg_type load_from_initializer(std::initializer_list<int8_t> init) {
        alignas(16) int8_t temp[16] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 16 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1q_s8(temp);
    }

    static void store(int8_t* ptr, reg_type reg) { vst1q_s8(ptr, reg); }
    static void store_aligned(int8_t* ptr, reg_type reg) { vst1q_s8(ptr, reg); }

    static int8_t extract(reg_type reg, size_t index) {
        alignas(16) int8_t temp[16];
        vst1q_s8(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vaddq_s8(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsubq_s8(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmulq_s8(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(16) int8_t temp_a[16], temp_b[16], result[16];
        vst1q_s8(temp_a, a);
        vst1q_s8(temp_b, b);
        for (size_t i = 0; i < 16; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1q_s8(result);
    }

    static reg_type neg(reg_type a) { return vnegq_s8(a); }

    static bool equal(reg_type a, reg_type b) {
        uint8x16_t result = vceqq_s8(a, b);
        uint16x8_t result16 = vpaddlq_u8(result);
        uint32x4_t result32 = vpaddlq_u16(result16);
        uint64x2_t result64 = vpaddlq_u32(result32);
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vminq_s8(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmaxq_s8(a, b); }
    static reg_type abs(reg_type a) { return vabsq_s8(a); }

    // Bitwise Operations
    static reg_type bitwise_and(reg_type a, reg_type b) { return vandq_s8(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return vorrq_s8(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return veorq_s8(a, b); }
    static reg_type bitwise_not(reg_type a) { return vmvnq_s8(a); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return vbicq_s8(a, b); }

    // Shift Operations
    static reg_type shift_left(reg_type a, int count) {
        return vshlq_s8(a, vdupq_n_s8(static_cast<int8_t>(count)));
    }
    static reg_type shift_right(reg_type a, int count) {
        return vshlq_s8(a, vdupq_n_s8(static_cast<int8_t>(-count)));
    }

    // Vector splitting operations
    static int8x8_t get_low(reg_type a) { return vget_low_s8(a); }
    static int8x8_t get_high(reg_type a) { return vget_high_s8(a); }
};

//=============================================================================
// NEON Backend Operations - fp16x8 (if supported)
//=============================================================================

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct backend_ops<neon_backend, fp16_t, 8> {
    using reg_type = float16x8_t;

    static reg_type zero() { return vdupq_n_f16(0.0f); }
    static reg_type set1(fp16_t scalar) { return vdupq_n_f16(scalar); }
    static reg_type load(const fp16_t* ptr) { return vld1q_f16(ptr); }
    static reg_type load_aligned(const fp16_t* ptr) { return vld1q_f16(ptr); }

    static reg_type load_from_initializer(std::initializer_list<fp16_t> init) {
        alignas(16) fp16_t temp[8] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1q_f16(temp);
    }

    static void store(fp16_t* ptr, reg_type reg) { vst1q_f16(ptr, reg); }
    static void store_aligned(fp16_t* ptr, reg_type reg) { vst1q_f16(ptr, reg); }

    static fp16_t extract(reg_type reg, size_t index) {
        alignas(16) fp16_t temp[8];
        vst1q_f16(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vaddq_f16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsubq_f16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmulq_f16(a, b); }
    static reg_type div(reg_type a, reg_type b) { return vdivq_f16(a, b); }
    static reg_type neg(reg_type a) { return vnegq_f16(a); }

    static bool equal(reg_type a, reg_type b) {
        uint16x8_t result = vceqq_f16(a, b);
        uint64x2_t result64 = vpaddlq_u32(vpaddlq_u16(result));
        uint64_t final_result = vgetq_lane_u64(result64, 0) + vgetq_lane_u64(result64, 1);
        return final_result == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vminq_f16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmaxq_f16(a, b); }
    static reg_type abs(reg_type a) { return vabsq_f16(a); }

    static reg_type fma(reg_type a, reg_type b, reg_type c) {
        return vfmaq_f16(c, a, b);
    }

    // Vector splitting operations
    static float16x4_t get_low(reg_type a) { return vget_low_f16(a); }
    static float16x4_t get_high(reg_type a) { return vget_high_f16(a); }
};
#endif

//=============================================================================
// NEON Backend Operations - Narrow Register Types (64-bit registers)
//=============================================================================
// These are needed for get_low/get_high operations

// uint8x8 (64-bit narrow register)
template<>
struct backend_ops<neon_backend, uint8_t, 8> {
    using reg_type = uint8x8_t;

    static reg_type zero() { return vdup_n_u8(0); }
    static reg_type set1(uint8_t scalar) { return vdup_n_u8(scalar); }
    static reg_type load(const uint8_t* ptr) { return vld1_u8(ptr); }
    static reg_type load_aligned(const uint8_t* ptr) { return vld1_u8(ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint8_t> init) {
        alignas(8) uint8_t temp[8] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1_u8(temp);
    }

    static void store(uint8_t* ptr, reg_type reg) { vst1_u8(ptr, reg); }
    static void store_aligned(uint8_t* ptr, reg_type reg) { vst1_u8(ptr, reg); }

    static uint8_t extract(reg_type reg, size_t index) {
        alignas(8) uint8_t temp[8];
        vst1_u8(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vadd_u8(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsub_u8(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmul_u8(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(8) uint8_t temp_a[8], temp_b[8], result[8];
        vst1_u8(temp_a, a);
        vst1_u8(temp_b, b);
        for (size_t i = 0; i < 8; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1_u8(result);
    }

    static reg_type neg(reg_type a) { return vsub_u8(vdup_n_u8(0), a); }

    static bool equal(reg_type a, reg_type b) {
        uint8x8_t result = vceq_u8(a, b);
        uint64_t result64 = vget_lane_u64(vreinterpret_u64_u8(result), 0);
        return result64 == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vmin_u8(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmax_u8(a, b); }
    static reg_type abs(reg_type a) { return a; }

    //-------------------------------------------------------------------------
    // Type Conversion Operations
    //-------------------------------------------------------------------------

    // Widening: uint8x8 → uint16x8
    static uint16x8_t convert_widen(reg_type a) {
        return vmovl_u8(a);  // Single NEON instruction (1 cycle)
    }

    // uint8 -> int8 (same width, reinterpret)
    static int8x8_t convert_to_signed(reg_type a) {
        return vreinterpret_s8_u8(a);  // 0 cycle (just type alias)
    }
};

// int8x8 (64-bit narrow register)
template<>
struct backend_ops<neon_backend, int8_t, 8> {
    using reg_type = int8x8_t;

    static reg_type zero() { return vdup_n_s8(0); }
    static reg_type set1(int8_t scalar) { return vdup_n_s8(scalar); }
    static reg_type load(const int8_t* ptr) { return vld1_s8(ptr); }
    static reg_type load_aligned(const int8_t* ptr) { return vld1_s8(ptr); }

    static reg_type load_from_initializer(std::initializer_list<int8_t> init) {
        alignas(8) int8_t temp[8] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1_s8(temp);
    }

    static void store(int8_t* ptr, reg_type reg) { vst1_s8(ptr, reg); }
    static void store_aligned(int8_t* ptr, reg_type reg) { vst1_s8(ptr, reg); }

    static int8_t extract(reg_type reg, size_t index) {
        alignas(8) int8_t temp[8];
        vst1_s8(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vadd_s8(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsub_s8(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmul_s8(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(8) int8_t temp_a[8], temp_b[8], result[8];
        vst1_s8(temp_a, a);
        vst1_s8(temp_b, b);
        for (size_t i = 0; i < 8; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1_s8(result);
    }

    static reg_type neg(reg_type a) { return vneg_s8(a); }

    static bool equal(reg_type a, reg_type b) {
        uint8x8_t result = vceq_s8(a, b);
        uint64_t result64 = vget_lane_u64(vreinterpret_u64_u8(result), 0);
        return result64 == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vmin_s8(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmax_s8(a, b); }
    static reg_type abs(reg_type a) { return vabs_s8(a); }

    //-------------------------------------------------------------------------
    // Type Conversion Operations
    //-------------------------------------------------------------------------

    // Widening: int8x8 → int16x8
    static int16x8_t convert_widen(reg_type a) {
        return vmovl_s8(a);  // Single NEON instruction (1 cycle)
    }
};

// uint16x4 (64-bit narrow register)
template<>
struct backend_ops<neon_backend, uint16_t, 4> {
    using reg_type = uint16x4_t;

    static reg_type zero() { return vdup_n_u16(0); }
    static reg_type set1(uint16_t scalar) { return vdup_n_u16(scalar); }
    static reg_type load(const uint16_t* ptr) { return vld1_u16(ptr); }
    static reg_type load_aligned(const uint16_t* ptr) { return vld1_u16(ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint16_t> init) {
        alignas(8) uint16_t temp[4] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1_u16(temp);
    }

    static void store(uint16_t* ptr, reg_type reg) { vst1_u16(ptr, reg); }
    static void store_aligned(uint16_t* ptr, reg_type reg) { vst1_u16(ptr, reg); }

    static uint16_t extract(reg_type reg, size_t index) {
        alignas(8) uint16_t temp[4];
        vst1_u16(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vadd_u16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsub_u16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmul_u16(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(8) uint16_t temp_a[4], temp_b[4], result[4];
        vst1_u16(temp_a, a);
        vst1_u16(temp_b, b);
        for (size_t i = 0; i < 4; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1_u16(result);
    }

    static reg_type neg(reg_type a) { return vsub_u16(vdup_n_u16(0), a); }

    static bool equal(reg_type a, reg_type b) {
        uint16x4_t result = vceq_u16(a, b);
        uint64_t result64 = vget_lane_u64(vreinterpret_u64_u16(result), 0);
        return result64 == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vmin_u16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmax_u16(a, b); }
    static reg_type abs(reg_type a) { return a; }

    // uint16 -> int16 (same width, reinterpret)
    static int16x4_t convert_to_signed(reg_type a) {
        return vreinterpret_s16_u16(a);  // 0 cycle (just type alias)
    }
};

// int16x4 (64-bit narrow register)
template<>
struct backend_ops<neon_backend, int16_t, 4> {
    using reg_type = int16x4_t;

    static reg_type zero() { return vdup_n_s16(0); }
    static reg_type set1(int16_t scalar) { return vdup_n_s16(scalar); }
    static reg_type load(const int16_t* ptr) { return vld1_s16(ptr); }
    static reg_type load_aligned(const int16_t* ptr) { return vld1_s16(ptr); }

    static reg_type load_from_initializer(std::initializer_list<int16_t> init) {
        alignas(8) int16_t temp[4] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1_s16(temp);
    }

    static void store(int16_t* ptr, reg_type reg) { vst1_s16(ptr, reg); }
    static void store_aligned(int16_t* ptr, reg_type reg) { vst1_s16(ptr, reg); }

    static int16_t extract(reg_type reg, size_t index) {
        alignas(8) int16_t temp[4];
        vst1_s16(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vadd_s16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsub_s16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmul_s16(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(8) int16_t temp_a[4], temp_b[4], result[4];
        vst1_s16(temp_a, a);
        vst1_s16(temp_b, b);
        for (size_t i = 0; i < 4; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1_s16(result);
    }

    static reg_type neg(reg_type a) { return vneg_s16(a); }

    static bool equal(reg_type a, reg_type b) {
        uint16x4_t result = vceq_s16(a, b);
        uint64_t result64 = vget_lane_u64(vreinterpret_u64_u16(result), 0);
        return result64 == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vmin_s16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmax_s16(a, b); }
    static reg_type abs(reg_type a) { return vabs_s16(a); }
};

// uint32x2 (64-bit narrow register)
template<>
struct backend_ops<neon_backend, uint32_t, 2> {
    using reg_type = uint32x2_t;

    static reg_type zero() { return vdup_n_u32(0); }
    static reg_type set1(uint32_t scalar) { return vdup_n_u32(scalar); }
    static reg_type load(const uint32_t* ptr) { return vld1_u32(ptr); }
    static reg_type load_aligned(const uint32_t* ptr) { return vld1_u32(ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint32_t> init) {
        alignas(8) uint32_t temp[2] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 2 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1_u32(temp);
    }

    static void store(uint32_t* ptr, reg_type reg) { vst1_u32(ptr, reg); }
    static void store_aligned(uint32_t* ptr, reg_type reg) { vst1_u32(ptr, reg); }

    static uint32_t extract(reg_type reg, size_t index) {
        alignas(8) uint32_t temp[2];
        vst1_u32(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vadd_u32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsub_u32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmul_u32(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(8) uint32_t temp_a[2], temp_b[2], result[2];
        vst1_u32(temp_a, a);
        vst1_u32(temp_b, b);
        for (size_t i = 0; i < 2; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1_u32(result);
    }

    static reg_type neg(reg_type a) { return vsub_u32(vdup_n_u32(0), a); }

    static bool equal(reg_type a, reg_type b) {
        uint32x2_t result = vceq_u32(a, b);
        uint64_t result64 = vget_lane_u64(vreinterpret_u64_u32(result), 0);
        return result64 == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vmin_u32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmax_u32(a, b); }
    static reg_type abs(reg_type a) { return a; }

    // uint32 -> int32 (same width, reinterpret)
    static int32x2_t convert_to_signed(reg_type a) {
        return vreinterpret_s32_u32(a);  // 0 cycle (just type alias)
    }
};

// int32x2 (64-bit narrow register)
template<>
struct backend_ops<neon_backend, int32_t, 2> {
    using reg_type = int32x2_t;

    static reg_type zero() { return vdup_n_s32(0); }
    static reg_type set1(int32_t scalar) { return vdup_n_s32(scalar); }
    static reg_type load(const int32_t* ptr) { return vld1_s32(ptr); }
    static reg_type load_aligned(const int32_t* ptr) { return vld1_s32(ptr); }

    static reg_type load_from_initializer(std::initializer_list<int32_t> init) {
        alignas(8) int32_t temp[2] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 2 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1_s32(temp);
    }

    static void store(int32_t* ptr, reg_type reg) { vst1_s32(ptr, reg); }
    static void store_aligned(int32_t* ptr, reg_type reg) { vst1_s32(ptr, reg); }

    static int32_t extract(reg_type reg, size_t index) {
        alignas(8) int32_t temp[2];
        vst1_s32(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vadd_s32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsub_s32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmul_s32(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(8) int32_t temp_a[2], temp_b[2], result[2];
        vst1_s32(temp_a, a);
        vst1_s32(temp_b, b);
        for (size_t i = 0; i < 2; ++i) {
            result[i] = temp_a[i] / temp_b[i];
        }
        return vld1_s32(result);
    }

    static reg_type neg(reg_type a) { return vneg_s32(a); }

    static bool equal(reg_type a, reg_type b) {
        uint32x2_t result = vceq_s32(a, b);
        uint64_t result64 = vget_lane_u64(vreinterpret_u64_u32(result), 0);
        return result64 == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vmin_s32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmax_s32(a, b); }
    static reg_type abs(reg_type a) { return vabs_s32(a); }
};

// fp16x4 (64-bit narrow register) - for fp16 <-> fp32 conversions
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template<>
struct backend_ops<neon_backend, fp16_t, 4> {
    using reg_type = float16x4_t;

    static reg_type zero() { return vdup_n_f16(0.0f); }
    static reg_type set1(fp16_t scalar) { return vdup_n_f16(scalar); }
    static reg_type load(const fp16_t* ptr) { return vld1_f16(ptr); }
    static reg_type load_aligned(const fp16_t* ptr) { return vld1_f16(ptr); }

    static reg_type load_from_initializer(std::initializer_list<fp16_t> init) {
        alignas(8) fp16_t temp[4] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return vld1_f16(temp);
    }

    static void store(fp16_t* ptr, reg_type reg) { vst1_f16(ptr, reg); }
    static void store_aligned(fp16_t* ptr, reg_type reg) { vst1_f16(ptr, reg); }

    static fp16_t extract(reg_type reg, size_t index) {
        alignas(8) fp16_t temp[4];
        vst1_f16(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return vadd_f16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return vsub_f16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return vmul_f16(a, b); }
    static reg_type div(reg_type a, reg_type b) { return vdiv_f16(a, b); }
    static reg_type neg(reg_type a) { return vneg_f16(a); }

    static bool equal(reg_type a, reg_type b) {
        uint16x4_t result = vceq_f16(a, b);
        uint64_t result64 = vget_lane_u64(vreinterpret_u64_u16(result), 0);
        return result64 == 0xFFFFFFFFFFFFFFFFULL;
    }

    static reg_type min(reg_type a, reg_type b) { return vmin_f16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return vmax_f16(a, b); }
    static reg_type abs(reg_type a) { return vabs_f16(a); }

    //-------------------------------------------------------------------------
    // Type Conversion Operations
    //-------------------------------------------------------------------------

    // fp16 -> fp32 (4个fp16 -> 4个fp32)
    static float32x4_t convert_to_fp32(reg_type a) {
        return vcvt_f32_f16(a);  // Single NEON instruction (3 cycles)
    }
};
#endif

} // namespace tiny_simd

#endif // TINY_SIMD_ARM_NEON

#endif // TINY_SIMD_NEON_HPP
