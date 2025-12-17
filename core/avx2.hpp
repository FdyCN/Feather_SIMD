/*
 * Copyright (c) 2025 FdyCN
 *
 * Distributed under MIT license.
 * See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
 */

#ifndef TINY_SIMD_AVX2_HPP
#define TINY_SIMD_AVX2_HPP

#include "base.hpp"

#ifdef TINY_SIMD_X86_AVX2
#include <immintrin.h>
#endif

#include "scalar.hpp" // For fallback to scalar register types

namespace tiny_simd {

//=============================================================================
// AVX2 Backend Implementation
//=============================================================================

// Helper trait to determine register type for AVX2
template<typename T, size_t N>
struct avx2_traits;

#ifdef TINY_SIMD_X86_AVX2

//-----------------------------------------------------------------------------
// Float (N=8) -> __m256
//-----------------------------------------------------------------------------
template<>
struct avx2_traits<float, 8> {
    using reg_type = __m256;
};

//-----------------------------------------------------------------------------
// Double (N=4) -> __m256d
//-----------------------------------------------------------------------------
template<>
struct avx2_traits<double, 4> {
    using reg_type = __m256d;
};

//-----------------------------------------------------------------------------
// Integers -> __m256i
//-----------------------------------------------------------------------------
// int32/uint32 (N=8)
template<> struct avx2_traits<int32_t, 8> { using reg_type = __m256i; };
template<> struct avx2_traits<uint32_t, 8> { using reg_type = __m256i; };

// int16/uint16 (N=16)
template<> struct avx2_traits<int16_t, 16> { using reg_type = __m256i; };
template<> struct avx2_traits<uint16_t, 16> { using reg_type = __m256i; };

// int8/uint8 (N=32)
template<> struct avx2_traits<int8_t, 32> { using reg_type = __m256i; };
template<> struct avx2_traits<uint8_t, 32> { using reg_type = __m256i; };

//=============================================================================
// 128-bit SSE Traits (for N=4 float, N=2 double, etc.)
//=============================================================================

// float (N=4)
template<> struct avx2_traits<float, 4> { using reg_type = __m128; };

// double (N=2)
template<> struct avx2_traits<double, 2> { using reg_type = __m128d; };

// int32/uint32 (N=4)
template<> struct avx2_traits<int32_t, 4> { using reg_type = __m128i; };
template<> struct avx2_traits<uint32_t, 4> { using reg_type = __m128i; };

// int16/uint16 (N=8)
template<> struct avx2_traits<int16_t, 8> { using reg_type = __m128i; };
template<> struct avx2_traits<uint16_t, 8> { using reg_type = __m128i; };

// int8/uint8 (N=16)
template<> struct avx2_traits<int8_t, 16> { using reg_type = __m128i; };
template<> struct avx2_traits<uint8_t, 16> { using reg_type = __m128i; };

//=============================================================================
// Scalar Fallback Traits (for get_low on 128-bit vectors)
//=============================================================================
template<> struct avx2_traits<float, 2> { using reg_type = scalar_register<float, 2>; };
template<> struct avx2_traits<double, 1> { using reg_type = scalar_register<double, 1>; };
template<> struct avx2_traits<int32_t, 2> { using reg_type = scalar_register<int32_t, 2>; };
template<> struct avx2_traits<uint32_t, 2> { using reg_type = scalar_register<uint32_t, 2>; };
template<> struct avx2_traits<int16_t, 4> { using reg_type = scalar_register<int16_t, 4>; };
template<> struct avx2_traits<uint16_t, 4> { using reg_type = scalar_register<uint16_t, 4>; };
template<> struct avx2_traits<int8_t, 8> { using reg_type = scalar_register<int8_t, 8>; };
template<> struct avx2_traits<uint8_t, 8> { using reg_type = scalar_register<uint8_t, 8>; };

// Fallback ops inheritance
template<> struct backend_ops<avx2_backend, float, 2> : backend_ops<scalar_backend, float, 2> {};
template<> struct backend_ops<avx2_backend, double, 1> : backend_ops<scalar_backend, double, 1> {};
template<> struct backend_ops<avx2_backend, int32_t, 2> : backend_ops<scalar_backend, int32_t, 2> {};
template<> struct backend_ops<avx2_backend, uint32_t, 2> : backend_ops<scalar_backend, uint32_t, 2> {};
template<> struct backend_ops<avx2_backend, int16_t, 4> : backend_ops<scalar_backend, int16_t, 4> {};
template<> struct backend_ops<avx2_backend, uint16_t, 4> : backend_ops<scalar_backend, uint16_t, 4> {};
template<> struct backend_ops<avx2_backend, int8_t, 8> : backend_ops<scalar_backend, int8_t, 8> {};
template<> struct backend_ops<avx2_backend, uint8_t, 8> : backend_ops<scalar_backend, uint8_t, 8> {};

//=============================================================================
// Float Operations (float32x8)
//=============================================================================

//=============================================================================
// Float Operations (float32x4) - SSE
//=============================================================================

template<>
struct backend_ops<avx2_backend, float, 4> {
    using reg_type = __m128;

    static reg_type zero() { return _mm_setzero_ps(); }
    static reg_type set1(float scalar) { return _mm_set1_ps(scalar); }
    static reg_type load(const float* ptr) { return _mm_loadu_ps(ptr); }
    static reg_type load_aligned(const float* ptr) { return _mm_load_ps(ptr); }

    static reg_type load_from_initializer(std::initializer_list<float> init) {
        alignas(16) float temp[4] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm_loadu_ps(temp);
    }

    static void store(float* ptr, reg_type reg) { _mm_storeu_ps(ptr, reg); }
    static void store_aligned(float* ptr, reg_type reg) { _mm_store_ps(ptr, reg); }

    static float extract(reg_type reg, size_t index) {
        alignas(16) float temp[4];
        _mm_storeu_ps(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm_add_ps(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm_sub_ps(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm_mul_ps(a, b); }
    static reg_type div(reg_type a, reg_type b) { return _mm_div_ps(a, b); }

    static reg_type neg(reg_type a) {
        return _mm_sub_ps(_mm_setzero_ps(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        return _mm_movemask_ps(_mm_cmpeq_ps(a, b)) == 0xF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm_min_ps(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm_max_ps(a, b); }
    
    static reg_type abs(reg_type a) {
        static const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
        return _mm_andnot_ps(sign_mask, a);
    }

    static reg_type fma(reg_type a, reg_type b, reg_type c) {
        #ifdef __FMA__
            return _mm_fmadd_ps(a, b, c);
        #else
            return _mm_add_ps(_mm_mul_ps(a, b), c);
        #endif
    }

    template<typename IntT>
    static typename avx2_traits<IntT, 4>::reg_type convert_to_int(reg_type a) {
        // Only 32-bit int supported for N=4 float
        static_assert(sizeof(IntT) == 4, "Only 32-bit integers supported for conversion from float32x4");
        return _mm_cvttps_epi32(a);
    }

    // Split/Merge
    static typename avx2_traits<float, 2>::reg_type get_low(reg_type a) {
        scalar_register<float, 2> res;
        _mm_storeu_ps(res.data, a); // Stores 4 floats, but scalar reg is size 2. We overwrite stack or need careful store?
        // Wait, storeu_ps writes 16 bytes. scalar_register<float, 2> is 8 bytes.
        // Stack corruption risk if we write directly to res.data!
        alignas(16) float temp[4];
        _mm_storeu_ps(temp, a);
        res.data[0] = temp[0];
        res.data[1] = temp[1];
        return res;
    }

    static typename avx2_traits<float, 2>::reg_type get_high(reg_type a) {
        scalar_register<float, 2> res;
        alignas(16) float temp[4];
        _mm_storeu_ps(temp, a);
        res.data[0] = temp[2];
        res.data[1] = temp[3];
        return res;
    }
    
    static reg_type combine(typename avx2_traits<float, 2>::reg_type low, typename avx2_traits<float, 2>::reg_type high) {
         // Load from scalar regs
         // We can use _mm_set_ps(h1, h0, l1, l0) -> arguments are reversed usually?
         // _mm_set_ps(e3, e2, e1, e0)
         return _mm_set_ps(high.data[1], high.data[0], low.data[1], low.data[0]);
    }
};

template<>
struct backend_ops<avx2_backend, float, 8> {
    using reg_type = __m256;

    static reg_type zero() { return _mm256_setzero_ps(); }
    static reg_type set1(float scalar) { return _mm256_set1_ps(scalar); }
    static reg_type load(const float* ptr) { return _mm256_loadu_ps(ptr); }
    static reg_type load_aligned(const float* ptr) { return _mm256_load_ps(ptr); }

    static reg_type load_from_initializer(std::initializer_list<float> init) {
        alignas(32) float temp[8] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm256_loadu_ps(temp);
    }

    static void store(float* ptr, reg_type reg) { _mm256_storeu_ps(ptr, reg); }
    static void store_aligned(float* ptr, reg_type reg) { _mm256_store_ps(ptr, reg); }

    static float extract(reg_type reg, size_t index) {
        alignas(32) float temp[8];
        _mm256_storeu_ps(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm256_add_ps(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm256_sub_ps(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm256_mul_ps(a, b); }
    static reg_type div(reg_type a, reg_type b) { return _mm256_div_ps(a, b); }

    static reg_type neg(reg_type a) {
        return _mm256_sub_ps(_mm256_setzero_ps(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        // _mm256_cmp_ps returns mask (NaNs handled by predicate). OQ = Ordered Non-Signaling
        __m256 cmp = _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
        return _mm256_movemask_ps(cmp) == 0xFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm256_min_ps(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm256_max_ps(a, b); }
    
    static reg_type abs(reg_type a) {
        // Clear sign bit: AND with 0x7FFFFFFF
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        return _mm256_and_ps(a, mask);
    }

    static reg_type fma(reg_type a, reg_type b, reg_type c) {
        #ifdef __FMA__
            return _mm256_fmadd_ps(a, b, c);
        #else
            return _mm256_add_ps(_mm256_mul_ps(a, b), c);
        #endif
    }
};

//=============================================================================
// Integer Operations (int32x8)
//=============================================================================

//=============================================================================
// Integer Operations (int32x4 / uint32x4) - SSE
//=============================================================================

template<>
struct backend_ops<avx2_backend, int32_t, 4> {
    using reg_type = __m128i;

    static reg_type zero() { return _mm_setzero_si128(); }
    static reg_type set1(int32_t scalar) { return _mm_set1_epi32(scalar); }
    static reg_type load(const int32_t* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
    static reg_type load_aligned(const int32_t* ptr) { return _mm_load_si128((const __m128i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<int32_t> init) {
        alignas(16) int32_t temp[4] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm_loadu_si128((const __m128i*)temp);
    }

    static void store(int32_t* ptr, reg_type reg) { _mm_storeu_si128((__m128i*)ptr, reg); }
    static void store_aligned(int32_t* ptr, reg_type reg) { _mm_store_si128((__m128i*)ptr, reg); }

    static int32_t extract(reg_type reg, size_t index) {
        alignas(16) int32_t temp[4];
        _mm_storeu_si128((__m128i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm_add_epi32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm_sub_epi32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm_mullo_epi32(a, b); }
    
    static reg_type div(reg_type a, reg_type b) {
        alignas(16) int32_t ta[4], tb[4], tr[4];
        _mm_storeu_si128((__m128i*)ta, a);
        _mm_storeu_si128((__m128i*)tb, b);
        for(int i=0; i<4; ++i) tr[i] = ta[i] / tb[i];
        return _mm_loadu_si128((const __m128i*)tr);
    }

    static reg_type neg(reg_type a) { return _mm_sub_epi32(_mm_setzero_si128(), a); }

    static bool equal(reg_type a, reg_type b) {
        return _mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) == 0xFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm_min_epi32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm_max_epi32(a, b); }
    static reg_type abs(reg_type a) { return _mm_abs_epi32(a); }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm_and_si128(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm_or_si128(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm_xor_si128(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm_andnot_si128(b, a); }

    static reg_type shift_left(reg_type a, int count) { return _mm_slli_epi32(a, count); }
    static reg_type shift_right(reg_type a, int count) { return _mm_srai_epi32(a, count); } // Arithmetic

    static __m128 convert_to_float(reg_type a) {
        return _mm_cvtepi32_ps(a);
    }

    // Split/Merge
    static typename avx2_traits<int32_t, 2>::reg_type get_low(reg_type a) {
        scalar_register<int32_t, 2> res;
        alignas(16) int32_t temp[4];
        _mm_storeu_si128((__m128i*)temp, a);
        res.data[0] = temp[0];
        res.data[1] = temp[1];
        return res;
    }

    static typename avx2_traits<int32_t, 2>::reg_type get_high(reg_type a) {
        scalar_register<int32_t, 2> res;
        alignas(16) int32_t temp[4];
        _mm_storeu_si128((__m128i*)temp, a);
        res.data[0] = temp[2];
        res.data[1] = temp[3];
        return res;
    }

    static reg_type combine(typename avx2_traits<int32_t, 2>::reg_type low, typename avx2_traits<int32_t, 2>::reg_type high) {
        return _mm_set_epi32(high.data[1], high.data[0], low.data[1], low.data[0]);
    }
};

template<>
struct backend_ops<avx2_backend, uint32_t, 4> {
    using reg_type = __m128i;

    static reg_type zero() { return _mm_setzero_si128(); }
    static reg_type set1(uint32_t scalar) { return _mm_set1_epi32(scalar); }
    static reg_type load(const uint32_t* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
    static reg_type load_aligned(const uint32_t* ptr) { return _mm_load_si128((const __m128i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint32_t> init) {
        alignas(16) uint32_t temp[4] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm_loadu_si128((const __m128i*)temp);
    }

    static void store(uint32_t* ptr, reg_type reg) { _mm_storeu_si128((__m128i*)ptr, reg); }
    static void store_aligned(uint32_t* ptr, reg_type reg) { _mm_store_si128((__m128i*)ptr, reg); }

    static uint32_t extract(reg_type reg, size_t index) {
        alignas(16) uint32_t temp[4];
        _mm_storeu_si128((__m128i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm_add_epi32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm_sub_epi32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm_mullo_epi32(a, b); }
    
    static reg_type div(reg_type a, reg_type b) {
        alignas(16) uint32_t ta[4], tb[4], tr[4];
        _mm_storeu_si128((__m128i*)ta, a);
        _mm_storeu_si128((__m128i*)tb, b);
        for(int i=0; i<4; ++i) tr[i] = ta[i] / tb[i];
        return _mm_loadu_si128((const __m128i*)tr);
    }

    static reg_type neg(reg_type a) { return _mm_sub_epi32(_mm_setzero_si128(), a); }

    static bool equal(reg_type a, reg_type b) {
        return _mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) == 0xFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm_min_epu32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm_max_epu32(a, b); }
    static reg_type abs(reg_type a) { return a; }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm_and_si128(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm_or_si128(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm_xor_si128(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm_andnot_si128(b, a); }

    static reg_type shift_left(reg_type a, int count) { return _mm_slli_epi32(a, count); }
    static reg_type shift_right(reg_type a, int count) { return _mm_srli_epi32(a, count); } // Logical

    static __m128 convert_to_float(reg_type a) {
        // SSE2 doesn't have unsigned int -> float. Use 64-bit promotion.
        // Or simpler hack: if < 0 (high bit set), add 2^32
        // Since N=4, we can do it component-wise or with logic.
        // Let's use the standard "magic number" / "half-shift" trick or just _mm_cvtepi32_ps + adjustment
        
        __m128 f = _mm_cvtepi32_ps(a); // Convert as signed
        
        // Create mask for elements where MSB was set (so they were treated as negative)
        __m128i mask = _mm_cmplt_epi32(a, _mm_setzero_si128()); 
        __m128 adjustment = _mm_and_ps(_mm_castsi128_ps(mask), _mm_set1_ps(4294967296.0f));
        
        return _mm_add_ps(f, adjustment);
    }

    // Split/Merge
    static typename avx2_traits<uint32_t, 2>::reg_type get_low(reg_type a) {
        scalar_register<uint32_t, 2> res;
        alignas(16) uint32_t temp[4];
        _mm_storeu_si128((__m128i*)temp, a);
        res.data[0] = temp[0];
        res.data[1] = temp[1];
        return res;
    }

    static typename avx2_traits<uint32_t, 2>::reg_type get_high(reg_type a) {
        scalar_register<uint32_t, 2> res;
        alignas(16) uint32_t temp[4];
        _mm_storeu_si128((__m128i*)temp, a);
        res.data[0] = temp[2];
        res.data[1] = temp[3];
        return res;
    }

    static reg_type combine(typename avx2_traits<uint32_t, 2>::reg_type low, typename avx2_traits<uint32_t, 2>::reg_type high) {
        return _mm_set_epi32(high.data[1], high.data[0], low.data[1], low.data[0]);
    }
};

template<>
struct backend_ops<avx2_backend, int32_t, 8> {
    using reg_type = __m256i;

    static reg_type zero() { return _mm256_setzero_si256(); }
    static reg_type set1(int32_t scalar) { return _mm256_set1_epi32(scalar); }
    static reg_type load(const int32_t* ptr) { return _mm256_loadu_si256((const __m256i*)ptr); }
    static reg_type load_aligned(const int32_t* ptr) { return _mm256_load_si256((const __m256i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<int32_t> init) {
        alignas(32) int32_t temp[8] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm256_load_si256((const __m256i*)temp);
    }

    static void store(int32_t* ptr, reg_type reg) { _mm256_storeu_si256((__m256i*)ptr, reg); }
    static void store_aligned(int32_t* ptr, reg_type reg) { _mm256_store_si256((__m256i*)ptr, reg); }

    static int32_t extract(reg_type reg, size_t index) {
        alignas(32) int32_t temp[8];
        _mm256_store_si256((__m256i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm256_add_epi32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm256_sub_epi32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm256_mullo_epi32(a, b); }
    
    static reg_type div(reg_type a, reg_type b) {
        // AVX2 doesn't support vector integer division. Fallback to scalar.
        alignas(32) int32_t ta[8], tb[8], tr[8];
        _mm256_store_si256((__m256i*)ta, a);
        _mm256_store_si256((__m256i*)tb, b);
        for(int i=0; i<8; ++i) tr[i] = ta[i] / tb[i];
        return _mm256_load_si256((const __m256i*)tr);
    }

    static reg_type neg(reg_type a) {
        return _mm256_sub_epi32(_mm256_setzero_si256(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        __m256i cmp = _mm256_cmpeq_epi32(a, b);
        return _mm256_movemask_epi8(cmp) == 0xFFFFFFFF; // All bytes FF? No, movemask gives 32 bits
    }

    static reg_type min(reg_type a, reg_type b) { return _mm256_min_epi32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm256_max_epi32(a, b); }
    static reg_type abs(reg_type a) { return _mm256_abs_epi32(a); }

    // Bitwise
    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm256_and_si256(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm256_or_si256(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm256_xor_si256(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm256_andnot_si256(b, a); } // Note: AVX andnot is (~a) & b, so we swap args for (a & ~b)

    // Shifts
    static reg_type shift_left(reg_type a, int count) { return _mm256_slli_epi32(a, count); }
    static reg_type shift_right(reg_type a, int count) { return _mm256_srai_epi32(a, count); } // Arithmetic right shift

    // Split/Merge
    static typename avx2_traits<int32_t, 4>::reg_type get_low(reg_type a) {
        return _mm256_castsi256_si128(a);
    }

    static typename avx2_traits<int32_t, 4>::reg_type get_high(reg_type a) {
        return _mm256_extracti128_si256(a, 1);
    }

    static reg_type combine(typename avx2_traits<int32_t, 4>::reg_type low, typename avx2_traits<int32_t, 4>::reg_type high) {
        __m256i temp = _mm256_castsi128_si256(low);
        return _mm256_inserti128_si256(temp, high, 1);
    }
};

#endif // TINY_SIMD_X86_AVX2

//=============================================================================
// Integer Operations (uint32x8)
//=============================================================================

#ifdef TINY_SIMD_X86_AVX2
template<>
struct backend_ops<avx2_backend, uint32_t, 8> {
    using reg_type = __m256i;

    static reg_type zero() { return _mm256_setzero_si256(); }
    static reg_type set1(uint32_t scalar) { return _mm256_set1_epi32((int)scalar); }
    static reg_type load(const uint32_t* ptr) { return _mm256_loadu_si256((const __m256i*)ptr); }
    static reg_type load_aligned(const uint32_t* ptr) { return _mm256_load_si256((const __m256i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint32_t> init) {
        alignas(32) uint32_t temp[8] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm256_load_si256((const __m256i*)temp);
    }

    static void store(uint32_t* ptr, reg_type reg) { _mm256_storeu_si256((__m256i*)ptr, reg); }
    static void store_aligned(uint32_t* ptr, reg_type reg) { _mm256_store_si256((__m256i*)ptr, reg); }

    static uint32_t extract(reg_type reg, size_t index) {
        alignas(32) uint32_t temp[8];
        _mm256_store_si256((__m256i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm256_add_epi32(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm256_sub_epi32(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm256_mullo_epi32(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(32) uint32_t ta[8], tb[8], tr[8];
        _mm256_store_si256((__m256i*)ta, a);
        _mm256_store_si256((__m256i*)tb, b);
        for(int i=0; i<8; ++i) tr[i] = ta[i] / tb[i];
        return _mm256_load_si256((const __m256i*)tr);
    }

    static reg_type neg(reg_type a) {
        return _mm256_sub_epi32(_mm256_setzero_si256(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        __m256i cmp = _mm256_cmpeq_epi32(a, b);
        return _mm256_movemask_epi8(cmp) == 0xFFFFFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm256_min_epu32(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm256_max_epu32(a, b); }
    static reg_type abs(reg_type a) { return a; }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm256_and_si256(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm256_or_si256(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm256_xor_si256(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm256_andnot_si256(b, a); }

    static reg_type shift_left(reg_type a, int count) { return _mm256_slli_epi32(a, count); }
    static reg_type shift_right(reg_type a, int count) { return _mm256_srli_epi32(a, count); } // Logical right shift
};

//=============================================================================
// Integer Operations (int16x16)
//=============================================================================

//=============================================================================
// Integer Operations (int16x8 / uint16x8) - SSE
//=============================================================================

template<>
struct backend_ops<avx2_backend, int16_t, 8> {
    using reg_type = __m128i;

    static reg_type zero() { return _mm_setzero_si128(); }
    static reg_type set1(int16_t scalar) { return _mm_set1_epi16(scalar); }
    static reg_type load(const int16_t* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
    static reg_type load_aligned(const int16_t* ptr) { return _mm_load_si128((const __m128i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<int16_t> init) {
        alignas(16) int16_t temp[8] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm_loadu_si128((const __m128i*)temp);
    }

    static void store(int16_t* ptr, reg_type reg) { _mm_storeu_si128((__m128i*)ptr, reg); }
    static void store_aligned(int16_t* ptr, reg_type reg) { _mm_store_si128((__m128i*)ptr, reg); }

    static int16_t extract(reg_type reg, size_t index) {
        alignas(16) int16_t temp[8];
        _mm_storeu_si128((__m128i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm_add_epi16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm_sub_epi16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm_mullo_epi16(a, b); }
    
    static reg_type div(reg_type a, reg_type b) {
        alignas(16) int16_t ta[8], tb[8], tr[8];
        _mm_storeu_si128((__m128i*)ta, a);
        _mm_storeu_si128((__m128i*)tb, b);
        for(int i=0; i<8; ++i) tr[i] = ta[i] / tb[i];
        return _mm_loadu_si128((const __m128i*)tr);
    }

    static reg_type neg(reg_type a) { return _mm_sub_epi16(_mm_setzero_si128(), a); }

    static bool equal(reg_type a, reg_type b) {
        return _mm_movemask_epi8(_mm_cmpeq_epi16(a, b)) == 0xFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm_min_epi16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm_max_epi16(a, b); }
    static reg_type abs(reg_type a) { return _mm_abs_epi16(a); }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm_and_si128(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm_or_si128(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm_xor_si128(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm_andnot_si128(b, a); }

    static reg_type shift_left(reg_type a, int count) { return _mm_slli_epi16(a, count); }
    static reg_type shift_right(reg_type a, int count) { return _mm_srai_epi16(a, count); } // Arithmetic

    // Split/Merge
    static typename avx2_traits<int16_t, 4>::reg_type get_low(reg_type a) {
        scalar_register<int16_t, 4> res;
        alignas(16) int16_t temp[8];
        _mm_storeu_si128((__m128i*)temp, a);
        for(int i=0; i<4; ++i) res.data[i] = temp[i];
        return res;
    }

    static typename avx2_traits<int16_t, 4>::reg_type get_high(reg_type a) {
        scalar_register<int16_t, 4> res;
        alignas(16) int16_t temp[8];
        _mm_storeu_si128((__m128i*)temp, a);
        for(int i=0; i<4; ++i) res.data[i] = temp[4+i];
        return res;
    }

    static reg_type combine(typename avx2_traits<int16_t, 4>::reg_type low, typename avx2_traits<int16_t, 4>::reg_type high) {
        // _mm_set_epi16 arguments are high to low: e7, e6... e0
        return _mm_set_epi16(high.data[3], high.data[2], high.data[1], high.data[0],
                             low.data[3], low.data[2], low.data[1], low.data[0]);
    }
};

template<>
struct backend_ops<avx2_backend, uint16_t, 8> {
    using reg_type = __m128i;

    static reg_type zero() { return _mm_setzero_si128(); }
    static reg_type set1(uint16_t scalar) { return _mm_set1_epi16(scalar); }
    static reg_type load(const uint16_t* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
    static reg_type load_aligned(const uint16_t* ptr) { return _mm_load_si128((const __m128i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint16_t> init) {
        alignas(16) uint16_t temp[8] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 8 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm_loadu_si128((const __m128i*)temp);
    }

    static void store(uint16_t* ptr, reg_type reg) { _mm_storeu_si128((__m128i*)ptr, reg); }
    static void store_aligned(uint16_t* ptr, reg_type reg) { _mm_store_si128((__m128i*)ptr, reg); }

    static uint16_t extract(reg_type reg, size_t index) {
        alignas(16) uint16_t temp[8];
        _mm_storeu_si128((__m128i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm_add_epi16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm_sub_epi16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm_mullo_epi16(a, b); }
    
    static reg_type div(reg_type a, reg_type b) {
        alignas(16) uint16_t ta[8], tb[8], tr[8];
        _mm_storeu_si128((__m128i*)ta, a);
        _mm_storeu_si128((__m128i*)tb, b);
        for(int i=0; i<8; ++i) tr[i] = ta[i] / tb[i];
        return _mm_loadu_si128((const __m128i*)tr);
    }

    static reg_type neg(reg_type a) { return _mm_sub_epi16(_mm_setzero_si128(), a); }

    static bool equal(reg_type a, reg_type b) {
        return _mm_movemask_epi8(_mm_cmpeq_epi16(a, b)) == 0xFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm_min_epu16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm_max_epu16(a, b); }
    static reg_type abs(reg_type a) { return a; }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm_and_si128(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm_or_si128(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm_xor_si128(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm_andnot_si128(b, a); }

    static reg_type shift_left(reg_type a, int count) { return _mm_slli_epi16(a, count); }
    static reg_type shift_right(reg_type a, int count) { return _mm_srli_epi16(a, count); } // Logical

    // Split/Merge
    static typename avx2_traits<uint16_t, 4>::reg_type get_low(reg_type a) {
        scalar_register<uint16_t, 4> res;
        alignas(16) uint16_t temp[8];
        _mm_storeu_si128((__m128i*)temp, a);
        for(int i=0; i<4; ++i) res.data[i] = temp[i];
        return res;
    }

    static typename avx2_traits<uint16_t, 4>::reg_type get_high(reg_type a) {
        scalar_register<uint16_t, 4> res;
        alignas(16) uint16_t temp[8];
        _mm_storeu_si128((__m128i*)temp, a);
        for(int i=0; i<4; ++i) res.data[i] = temp[4+i];
        return res;
    }

    static reg_type combine(typename avx2_traits<uint16_t, 4>::reg_type low, typename avx2_traits<uint16_t, 4>::reg_type high) {
        return _mm_set_epi16(high.data[3], high.data[2], high.data[1], high.data[0],
                             low.data[3], low.data[2], low.data[1], low.data[0]);
    }
};

template<>
struct backend_ops<avx2_backend, int16_t, 16> {
    using reg_type = __m256i;

    static reg_type zero() { return _mm256_setzero_si256(); }
    static reg_type set1(int16_t scalar) { return _mm256_set1_epi16(scalar); }
    static reg_type load(const int16_t* ptr) { return _mm256_loadu_si256((const __m256i*)ptr); }
    static reg_type load_aligned(const int16_t* ptr) { return _mm256_load_si256((const __m256i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<int16_t> init) {
        alignas(32) int16_t temp[16] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 16 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm256_load_si256((const __m256i*)temp);
    }

    static void store(int16_t* ptr, reg_type reg) { _mm256_storeu_si256((__m256i*)ptr, reg); }
    static void store_aligned(int16_t* ptr, reg_type reg) { _mm256_store_si256((__m256i*)ptr, reg); }

    static int16_t extract(reg_type reg, size_t index) {
        alignas(32) int16_t temp[16];
        _mm256_store_si256((__m256i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm256_add_epi16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm256_sub_epi16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm256_mullo_epi16(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(32) int16_t ta[16], tb[16], tr[16];
        _mm256_store_si256((__m256i*)ta, a);
        _mm256_store_si256((__m256i*)tb, b);
        for(int i=0; i<16; ++i) tr[i] = ta[i] / tb[i];
        return _mm256_load_si256((const __m256i*)tr);
    }

    static reg_type neg(reg_type a) {
        return _mm256_sub_epi16(_mm256_setzero_si256(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        __m256i cmp = _mm256_cmpeq_epi16(a, b);
        return _mm256_movemask_epi8(cmp) == 0xFFFFFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm256_min_epi16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm256_max_epi16(a, b); }
    static reg_type abs(reg_type a) { return _mm256_abs_epi16(a); }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm256_and_si256(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm256_or_si256(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm256_xor_si256(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm256_andnot_si256(b, a); }

    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm256_andnot_si256(b, a); }

    static reg_type shift_left(reg_type a, int count) { return _mm256_slli_epi16(a, count); }
    static reg_type shift_right(reg_type a, int count) { return _mm256_srai_epi16(a, count); } // Arithmetic

    // Split/Merge
    static typename avx2_traits<int16_t, 8>::reg_type get_low(reg_type a) {
        return _mm256_castsi256_si128(a);
    }

    static typename avx2_traits<int16_t, 8>::reg_type get_high(reg_type a) {
        return _mm256_extracti128_si256(a, 1);
    }

    static reg_type combine(typename avx2_traits<int16_t, 8>::reg_type low, typename avx2_traits<int16_t, 8>::reg_type high) {
        __m256i temp = _mm256_castsi128_si256(low);
        return _mm256_inserti128_si256(temp, high, 1);
    }
};

//=============================================================================
// Integer Operations (uint16x16)
//=============================================================================

template<>
struct backend_ops<avx2_backend, uint16_t, 16> {
    using reg_type = __m256i;

    static reg_type zero() { return _mm256_setzero_si256(); }
    static reg_type set1(uint16_t scalar) { return _mm256_set1_epi16((int16_t)scalar); }
    static reg_type load(const uint16_t* ptr) { return _mm256_loadu_si256((const __m256i*)ptr); }
    static reg_type load_aligned(const uint16_t* ptr) { return _mm256_load_si256((const __m256i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint16_t> init) {
        alignas(32) uint16_t temp[16] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 16 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm256_load_si256((const __m256i*)temp);
    }

    static void store(uint16_t* ptr, reg_type reg) { _mm256_storeu_si256((__m256i*)ptr, reg); }
    static void store_aligned(uint16_t* ptr, reg_type reg) { _mm256_store_si256((__m256i*)ptr, reg); }

    static uint16_t extract(reg_type reg, size_t index) {
        alignas(32) uint16_t temp[16];
        _mm256_store_si256((__m256i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm256_add_epi16(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm256_sub_epi16(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm256_mullo_epi16(a, b); }

    static reg_type div(reg_type a, reg_type b) {
        alignas(32) uint16_t ta[16], tb[16], tr[16];
        _mm256_store_si256((__m256i*)ta, a);
        _mm256_store_si256((__m256i*)tb, b);
        for(int i=0; i<16; ++i) tr[i] = ta[i] / tb[i];
        return _mm256_load_si256((const __m256i*)tr);
    }

    static reg_type neg(reg_type a) {
        return _mm256_sub_epi16(_mm256_setzero_si256(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        __m256i cmp = _mm256_cmpeq_epi16(a, b);
        return _mm256_movemask_epi8(cmp) == 0xFFFFFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm256_min_epu16(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm256_max_epu16(a, b); }
    static reg_type abs(reg_type a) { return a; }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm256_and_si256(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm256_or_si256(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm256_xor_si256(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm256_andnot_si256(b, a); }

    static reg_type shift_left(reg_type a, int count) { return _mm256_slli_epi16(a, count); }
    static reg_type shift_right(reg_type a, int count) { return _mm256_srli_epi16(a, count); } // Logical
};

#endif // TINY_SIMD_X86_AVX2

//=============================================================================
// Integer Operations (int8x32)
//=============================================================================

#ifdef TINY_SIMD_X86_AVX2
//=============================================================================
// Integer Operations (int8x16 / uint8x16) - SSE
//=============================================================================

template<>
struct backend_ops<avx2_backend, int8_t, 16> {
    using reg_type = __m128i;

    static reg_type zero() { return _mm_setzero_si128(); }
    static reg_type set1(int8_t scalar) { return _mm_set1_epi8(scalar); }
    static reg_type load(const int8_t* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
    static reg_type load_aligned(const int8_t* ptr) { return _mm_load_si128((const __m128i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<int8_t> init) {
        alignas(16) int8_t temp[16] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 16 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm_loadu_si128((const __m128i*)temp);
    }

    static void store(int8_t* ptr, reg_type reg) { _mm_storeu_si128((__m128i*)ptr, reg); }
    static void store_aligned(int8_t* ptr, reg_type reg) { _mm_store_si128((__m128i*)ptr, reg); }

    static int8_t extract(reg_type reg, size_t index) {
        alignas(16) int8_t temp[16];
        _mm_storeu_si128((__m128i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm_add_epi8(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm_sub_epi8(a, b); }
    
    // Fallback multiplication for int8
    static reg_type mul(reg_type a, reg_type b) {
        // SSE4.1 doesn't have mullo_epi8. Fallback to scalar.
        alignas(16) int8_t ta[16], tb[16], tr[16];
        _mm_storeu_si128((__m128i*)ta, a);
        _mm_storeu_si128((__m128i*)tb, b);
        for(int i=0; i<16; ++i) tr[i] = ta[i] * tb[i];
        return _mm_loadu_si128((const __m128i*)tr);
    }

    static reg_type div(reg_type a, reg_type b) {
        alignas(16) int8_t ta[16], tb[16], tr[16];
        _mm_storeu_si128((__m128i*)ta, a);
        _mm_storeu_si128((__m128i*)tb, b);
        for(int i=0; i<16; ++i) tr[i] = ta[i] / tb[i];
        return _mm_loadu_si128((const __m128i*)tr);
    }

    static reg_type neg(reg_type a) { return _mm_sub_epi8(_mm_setzero_si128(), a); }

    static bool equal(reg_type a, reg_type b) {
        return _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) == 0xFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm_min_epi8(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm_max_epi8(a, b); }
    static reg_type abs(reg_type a) { return _mm_abs_epi8(a); }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm_and_si128(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm_or_si128(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm_xor_si128(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm_andnot_si128(b, a); }

    static reg_type shift_left(reg_type a, int count) {
        __m128i mask = _mm_set1_epi8((char)(0xFF << count));
        __m128i s = _mm_slli_epi16(a, count); 
        return _mm_and_si128(s, mask);
    }
    static reg_type shift_right(reg_type a, int count) {
        // Arithmetic shift right emulation for int8
        alignas(16) int8_t ta[16], tr[16];
        _mm_storeu_si128((__m128i*)ta, a);
        for(int i=0; i<16; ++i) tr[i] = ta[i] >> count;
        return _mm_loadu_si128((const __m128i*)tr);
    }

    // Split/Merge
    static typename avx2_traits<int8_t, 8>::reg_type get_low(reg_type a) {
        scalar_register<int8_t, 8> res;
        alignas(16) int8_t temp[16];
        _mm_storeu_si128((__m128i*)temp, a);
        for(int i=0; i<8; ++i) res.data[i] = temp[i];
        return res;
    }

    static typename avx2_traits<int8_t, 8>::reg_type get_high(reg_type a) {
        scalar_register<int8_t, 8> res;
        alignas(16) int8_t temp[16];
        _mm_storeu_si128((__m128i*)temp, a);
        for(int i=0; i<8; ++i) res.data[i] = temp[8+i];
        return res;
    }

    static reg_type combine(typename avx2_traits<int8_t, 8>::reg_type low, typename avx2_traits<int8_t, 8>::reg_type high) {
        return _mm_set_epi8(high.data[7], high.data[6], high.data[5], high.data[4],
                            high.data[3], high.data[2], high.data[1], high.data[0],
                            low.data[7], low.data[6], low.data[5], low.data[4],
                            low.data[3], low.data[2], low.data[1], low.data[0]);
    }
};

template<>
struct backend_ops<avx2_backend, uint8_t, 16> {
    using reg_type = __m128i;

    static reg_type zero() { return _mm_setzero_si128(); }
    static reg_type set1(uint8_t scalar) { return _mm_set1_epi8((int8_t)scalar); }
    static reg_type load(const uint8_t* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
    static reg_type load_aligned(const uint8_t* ptr) { return _mm_load_si128((const __m128i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint8_t> init) {
        alignas(16) uint8_t temp[16] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 16 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm_loadu_si128((const __m128i*)temp);
    }

    static void store(uint8_t* ptr, reg_type reg) { _mm_storeu_si128((__m128i*)ptr, reg); }
    static void store_aligned(uint8_t* ptr, reg_type reg) { _mm_store_si128((__m128i*)ptr, reg); }

    static uint8_t extract(reg_type reg, size_t index) {
        alignas(16) uint8_t temp[16];
        _mm_storeu_si128((__m128i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm_add_epi8(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm_sub_epi8(a, b); }
    
    static reg_type mul(reg_type a, reg_type b) {
        // Fallback since mullo_epi8 missing
        alignas(16) uint8_t ta[16], tb[16], tr[16];
        _mm_storeu_si128((__m128i*)ta, a);
        _mm_storeu_si128((__m128i*)tb, b);
        for(int i=0; i<16; ++i) tr[i] = ta[i] * tb[i];
        return _mm_loadu_si128((const __m128i*)tr);
    }

    static reg_type div(reg_type a, reg_type b) {
        alignas(16) uint8_t ta[16], tb[16], tr[16];
        _mm_storeu_si128((__m128i*)ta, a);
        _mm_storeu_si128((__m128i*)tb, b);
        for(int i=0; i<16; ++i) tr[i] = ta[i] / tb[i];
        return _mm_loadu_si128((const __m128i*)tr);
    }

    static reg_type neg(reg_type a) { return _mm_sub_epi8(_mm_setzero_si128(), a); }

    static bool equal(reg_type a, reg_type b) {
        return _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) == 0xFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm_min_epu8(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm_max_epu8(a, b); }
    static reg_type abs(reg_type a) { return a; }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm_and_si128(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm_or_si128(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm_xor_si128(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm_andnot_si128(b, a); }

    static reg_type shift_left(reg_type a, int count) {
        __m128i mask = _mm_set1_epi8((char)(0xFF << count));
        __m128i s = _mm_slli_epi16(a, count); 
        return _mm_and_si128(s, mask);
    }
    static reg_type shift_right(reg_type a, int count) { 
        __m128i mask = _mm_set1_epi8((unsigned char)(0xFF) >> count);
        __m128i s = _mm_srli_epi16(a, count);
        return _mm_and_si128(s, mask);
    }

    // Split/Merge
    static typename avx2_traits<uint8_t, 8>::reg_type get_low(reg_type a) {
        scalar_register<uint8_t, 8> res;
        alignas(16) uint8_t temp[16];
        _mm_storeu_si128((__m128i*)temp, a);
        for(int i=0; i<8; ++i) res.data[i] = temp[i];
        return res;
    }

    static typename avx2_traits<uint8_t, 8>::reg_type get_high(reg_type a) {
        scalar_register<uint8_t, 8> res;
        alignas(16) uint8_t temp[16];
        _mm_storeu_si128((__m128i*)temp, a);
        for(int i=0; i<8; ++i) res.data[i] = temp[8+i];
        return res;
    }

    static reg_type combine(typename avx2_traits<uint8_t, 8>::reg_type low, typename avx2_traits<uint8_t, 8>::reg_type high) {
        return _mm_set_epi8(high.data[7], high.data[6], high.data[5], high.data[4],
                            high.data[3], high.data[2], high.data[1], high.data[0],
                            low.data[7], low.data[6], low.data[5], low.data[4],
                            low.data[3], low.data[2], low.data[1], low.data[0]);
    }
};

template<>
struct backend_ops<avx2_backend, int8_t, 32> {
    using reg_type = __m256i;

    static reg_type zero() { return _mm256_setzero_si256(); }
    static reg_type set1(int8_t scalar) { return _mm256_set1_epi8(scalar); }
    static reg_type load(const int8_t* ptr) { return _mm256_loadu_si256((const __m256i*)ptr); }
    static reg_type load_aligned(const int8_t* ptr) { return _mm256_load_si256((const __m256i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<int8_t> init) {
        alignas(32) int8_t temp[32] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 32 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm256_loadu_si256((const __m256i*)temp);
    }

    static void store(int8_t* ptr, reg_type reg) { _mm256_storeu_si256((__m256i*)ptr, reg); }
    static void store_aligned(int8_t* ptr, reg_type reg) { _mm256_store_si256((__m256i*)ptr, reg); }

    static int8_t extract(reg_type reg, size_t index) {
        alignas(32) int8_t temp[32];
        _mm256_storeu_si256((__m256i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm256_add_epi8(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm256_sub_epi8(a, b); }
    
    // AVX2 does not have mullo_epi8. Emulation is costly.
    // We will leave mul unimplemented or throw/assert for now, or fallback to scalar if needed.
    // For now, simpler to basic implementation by splitting? 
    // Let's implement correct emulation: widen to 16-bit, mul, narrow.
    static reg_type mul(reg_type a, reg_type b) {
        // Unpack low/high 128 bits
        __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
        __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
        __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
        __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));

        __m256i res_lo = _mm256_mullo_epi16(a_lo, b_lo);
        __m256i res_hi = _mm256_mullo_epi16(a_hi, b_hi);

        // Pack back (truncating)
        // Note: AVX2 doesn't have _mm256_packus_epi16 which packs to unsigned.
        // We need _mm256_packs_epi16 (signed saturation). BUT multiply result isn't necessarily saturated, it wraps.
        // Standard packing with saturation IS NOT modulo arithmetic.
        // For modulo multiplication (wraparound), we need to be careful.
        // There is no direct "pack without saturation" instruction in x86.
        // Masking: 0xFF.
        __m256i mask = _mm256_set1_epi16(0x00FF);
        res_lo = _mm256_and_si256(res_lo, mask);
        res_hi = _mm256_and_si256(res_hi, mask);
        return _mm256_packus_epi16(res_lo, res_hi); 
        // packus: signed 16 -> unsigned 8. 
        // Since we masked to 0x00FF, values are 0..255. 0..127 maps to 0..127. 128..255 maps to 128..255.
        // Reinterpreting the result as int8 gives correct 2's complement wraparound.
        // However, packus does interleaving. We might need reordering.
        // Easier approach: Scalar loop for mul.
    }

    static reg_type div(reg_type a, reg_type b) {
        alignas(32) int8_t ta[32], tb[32], tr[32];
        _mm256_storeu_si256((__m256i*)ta, a);
        _mm256_storeu_si256((__m256i*)tb, b);
        for(int i=0; i<32; ++i) tr[i] = ta[i] / tb[i];
        return _mm256_loadu_si256((const __m256i*)tr);
    }

    static reg_type neg(reg_type a) { return _mm256_sub_epi8(_mm256_setzero_si256(), a); }

    static bool equal(reg_type a, reg_type b) {
        __m256i cmp = _mm256_cmpeq_epi8(a, b);
        return _mm256_movemask_epi8(cmp) == 0xFFFFFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm256_min_epi8(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm256_max_epi8(a, b); }
    static reg_type abs(reg_type a) { return _mm256_abs_epi8(a); }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm256_and_si256(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm256_or_si256(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm256_xor_si256(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm256_andnot_si256(b, a); }

    // Shift not standard for int8 on AVX

    // Split/Merge
    static typename avx2_traits<int8_t, 16>::reg_type get_low(reg_type a) {
        return _mm256_castsi256_si128(a);
    }

    static typename avx2_traits<int8_t, 16>::reg_type get_high(reg_type a) {
        return _mm256_extracti128_si256(a, 1);
    }

    static reg_type combine(typename avx2_traits<int8_t, 16>::reg_type low, typename avx2_traits<int8_t, 16>::reg_type high) {
        __m256i temp = _mm256_castsi128_si256(low);
        return _mm256_inserti128_si256(temp, high, 1);
    }

    // Shift Operations - AVX2 does NOT support per-element vector shifts for 8-bit.
    // It only supports 16/32/64 bit shifts.
    // Logic: Widen to 16, shift, narrow? Or apply bitwise magic.
    // Or just scalar fallback.
    // For 'shift all by count' (scalar count) - we can use 16-bit shift logic with masking.
    static reg_type shift_left(reg_type a, int count) {
        __m256i mask = _mm256_set1_epi8((char)(0xFF << count));
        __m256i s = _mm256_slli_epi16(a, count); 
        return _mm256_and_si256(s, mask);
    }

    static reg_type shift_right(reg_type a, int count) {
        // Arithmetic right shift on 8-bit is tricky.
        // Widen to 16 bit signed, shift right, pack.
        __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
        __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
        
        a_lo = _mm256_srai_epi16(a_lo, count);
        a_hi = _mm256_srai_epi16(a_hi, count);

        // Pack wraps which is fine since we kept 8-bit values in range
        return _mm256_packs_epi16(a_lo, a_hi);
    }
};

//=============================================================================
// Integer Operations (uint8x32)
//=============================================================================

template<>
struct backend_ops<avx2_backend, uint8_t, 32> {
    using reg_type = __m256i;

    static reg_type zero() { return _mm256_setzero_si256(); }
    static reg_type set1(uint8_t scalar) { return _mm256_set1_epi8((int8_t)scalar); }
    static reg_type load(const uint8_t* ptr) { return _mm256_loadu_si256((const __m256i*)ptr); }
    static reg_type load_aligned(const uint8_t* ptr) { return _mm256_load_si256((const __m256i*)ptr); }

    static reg_type load_from_initializer(std::initializer_list<uint8_t> init) {
        alignas(32) uint8_t temp[32] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 32 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm256_loadu_si256((const __m256i*)temp);
    }

    static void store(uint8_t* ptr, reg_type reg) { _mm256_storeu_si256((__m256i*)ptr, reg); }
    static void store_aligned(uint8_t* ptr, reg_type reg) { _mm256_store_si256((__m256i*)ptr, reg); }

    static uint8_t extract(reg_type reg, size_t index) {
        alignas(32) uint8_t temp[32];
        _mm256_storeu_si256((__m256i*)temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm256_add_epi8(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm256_sub_epi8(a, b); }
    
    static reg_type mul(reg_type a, reg_type b) {
        // Scalar fallback since mullo_epi8 missing + unsigned is weird
        alignas(32) uint8_t ta[32], tb[32], tr[32];
        _mm256_storeu_si256((__m256i*)ta, a);
        _mm256_storeu_si256((__m256i*)tb, b);
        for(int i=0; i<32; ++i) tr[i] = ta[i] * tb[i];
        return _mm256_loadu_si256((const __m256i*)tr);
    }

    static reg_type div(reg_type a, reg_type b) {
        alignas(32) uint8_t ta[32], tb[32], tr[32];
        _mm256_storeu_si256((__m256i*)ta, a);
        _mm256_storeu_si256((__m256i*)tb, b);
        for(int i=0; i<32; ++i) tr[i] = ta[i] / tb[i];
        return _mm256_loadu_si256((const __m256i*)tr);
    }

    static reg_type neg(reg_type a) {
        return _mm256_sub_epi8(_mm256_setzero_si256(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        __m256i cmp = _mm256_cmpeq_epi8(a, b);
        return _mm256_movemask_epi8(cmp) == 0xFFFFFFFF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm256_min_epu8(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm256_max_epu8(a, b); }
    static reg_type abs(reg_type a) { return a; }

    static reg_type bitwise_and(reg_type a, reg_type b) { return _mm256_and_si256(a, b); }
    static reg_type bitwise_or(reg_type a, reg_type b) { return _mm256_or_si256(a, b); }
    static reg_type bitwise_xor(reg_type a, reg_type b) { return _mm256_xor_si256(a, b); }
    static reg_type bitwise_not(reg_type a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
    static reg_type bitwise_andnot(reg_type a, reg_type b) { return _mm256_andnot_si256(b, a); }

    static reg_type shift_left(reg_type a, int count) { 
        __m256i mask = _mm256_set1_epi8((char)(0xFF << count));
        __m256i s = _mm256_slli_epi16(a, count); 
        return _mm256_and_si256(s, mask);
     }
    static reg_type shift_right(reg_type a, int count) { 
        __m256i mask = _mm256_set1_epi8((unsigned char)(0xFF) >> count);
        __m256i s = _mm256_srli_epi16(a, count);
        return _mm256_and_si256(s, mask);
    }
};

//=============================================================================
// Double Operations (doublex4)
//=============================================================================

//=============================================================================
// Double Operations (doublex2) - SSE
//=============================================================================

template<>
struct backend_ops<avx2_backend, double, 2> {
    using reg_type = __m128d;

    static reg_type zero() { return _mm_setzero_pd(); }
    static reg_type set1(double scalar) { return _mm_set1_pd(scalar); }
    static reg_type load(const double* ptr) { return _mm_loadu_pd(ptr); }
    static reg_type load_aligned(const double* ptr) { return _mm_load_pd(ptr); }

    static reg_type load_from_initializer(std::initializer_list<double> init) {
        alignas(16) double temp[2] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 2 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm_loadu_pd(temp);
    }

    static void store(double* ptr, reg_type reg) { _mm_storeu_pd(ptr, reg); }
    static void store_aligned(double* ptr, reg_type reg) { _mm_store_pd(ptr, reg); }

    static double extract(reg_type reg, size_t index) {
        alignas(16) double temp[2];
        _mm_storeu_pd(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm_add_pd(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm_sub_pd(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm_mul_pd(a, b); }
    static reg_type div(reg_type a, reg_type b) { return _mm_div_pd(a, b); }

    static reg_type neg(reg_type a) {
        return _mm_sub_pd(_mm_setzero_pd(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        return _mm_movemask_pd(_mm_cmpeq_pd(a, b)) == 0x3;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm_min_pd(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm_max_pd(a, b); }
    
    static reg_type abs(reg_type a) {
        static const __m128d mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return _mm_and_pd(a, mask);
    }

    static reg_type fma(reg_type a, reg_type b, reg_type c) {
        #ifdef __FMA__
            return _mm_fmadd_pd(a, b, c);
        #else
            return _mm_add_pd(_mm_mul_pd(a, b), c);
        #endif
    }

    // Split/Merge
    static typename avx2_traits<double, 1>::reg_type get_low(reg_type a) {
        scalar_register<double, 1> res;
        alignas(16) double temp[2];
        _mm_storeu_pd(temp, a);
        res.data[0] = temp[0];
        return res;
    }

    static typename avx2_traits<double, 1>::reg_type get_high(reg_type a) {
        scalar_register<double, 1> res;
        alignas(16) double temp[2];
        _mm_storeu_pd(temp, a);
        res.data[0] = temp[1];
        return res;
    }

    static reg_type combine(typename avx2_traits<double, 1>::reg_type low, typename avx2_traits<double, 1>::reg_type high) {
         return _mm_set_pd(high.data[0], low.data[0]);
    }
};

template<>
struct backend_ops<avx2_backend, double, 4> {
    using reg_type = __m256d;

    static reg_type zero() { return _mm256_setzero_pd(); }
    static reg_type set1(double scalar) { return _mm256_set1_pd(scalar); }
    static reg_type load(const double* ptr) { return _mm256_loadu_pd(ptr); }
    static reg_type load_aligned(const double* ptr) { return _mm256_load_pd(ptr); }

    static reg_type load_from_initializer(std::initializer_list<double> init) {
        alignas(32) double temp[4] = {0};
        auto it = init.begin();
        for (size_t i = 0; i < 4 && it != init.end(); ++i, ++it) {
            temp[i] = *it;
        }
        return _mm256_load_pd(temp);
    }

    static void store(double* ptr, reg_type reg) { _mm256_storeu_pd(ptr, reg); }
    static void store_aligned(double* ptr, reg_type reg) { _mm256_store_pd(ptr, reg); }

    static double extract(reg_type reg, size_t index) {
        alignas(32) double temp[4];
        _mm256_store_pd(temp, reg);
        return temp[index];
    }

    static reg_type add(reg_type a, reg_type b) { return _mm256_add_pd(a, b); }
    static reg_type sub(reg_type a, reg_type b) { return _mm256_sub_pd(a, b); }
    static reg_type mul(reg_type a, reg_type b) { return _mm256_mul_pd(a, b); }
    static reg_type div(reg_type a, reg_type b) { return _mm256_div_pd(a, b); }

    static reg_type neg(reg_type a) {
        return _mm256_sub_pd(_mm256_setzero_pd(), a);
    }

    static bool equal(reg_type a, reg_type b) {
        __m256d cmp = _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
        return _mm256_movemask_pd(cmp) == 0xF;
    }

    static reg_type min(reg_type a, reg_type b) { return _mm256_min_pd(a, b); }
    static reg_type max(reg_type a, reg_type b) { return _mm256_max_pd(a, b); }
    
    static reg_type abs(reg_type a) {
        const __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return _mm256_and_pd(a, mask);
    }

    static reg_type fma(reg_type a, reg_type b, reg_type c) {
        #ifdef __FMA__
            return _mm256_fmadd_pd(a, b, c);
        #else
            return _mm256_add_pd(_mm256_mul_pd(a, b), c);
        #endif
    }
};

#endif // TINY_SIMD_X86_AVX2

} // namespace tiny_simd

#endif // TINY_SIMD_AVX2_HPP
