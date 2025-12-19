#include <gtest/gtest.h>
#include "core/tiny_simd.hpp"

using namespace tiny_simd;

//=============================================================================
// Phase 3 Sign Conversion Tests: unsigned → signed
//=============================================================================

//-----------------------------------------------------------------------------
// Test 1: Same-width conversions (uint → int, same bit pattern)
//-----------------------------------------------------------------------------

TEST(SignConversionTest, UInt32ToInt32_SameWidth) {
    vec<uint32_t, 4> u32_data{0, 1, 2147483647u, 2147483648u};
    vec<int32_t, 4> s32_result = convert_to_signed(u32_data);

    // Bit pattern preserved
    EXPECT_EQ(s32_result[0], 0);
    EXPECT_EQ(s32_result[1], 1);
    EXPECT_EQ(s32_result[2], 2147483647);   // Max positive int32
    EXPECT_EQ(s32_result[3], static_cast<int32_t>(-2147483648LL));  // Wraps to min negative

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s32_result.is_simd_optimized);
    #endif
}

TEST(SignConversionTest, UInt32ToInt32_AllNegative) {
    vec<uint32_t, 4> u32_data{2147483648u, 3000000000u, 4000000000u, 4294967295u};
    vec<int32_t, 4> s32_result = convert_to_signed(u32_data);

    // All values > INT32_MAX wrap to negative
    EXPECT_EQ(s32_result[0], static_cast<int32_t>(-2147483648LL));
    EXPECT_LT(s32_result[1], 0);  // Negative
    EXPECT_LT(s32_result[2], 0);  // Negative
    EXPECT_EQ(s32_result[3], -1);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s32_result.is_simd_optimized);
    #endif
}

TEST(SignConversionTest, UInt16ToInt16_SameWidth) {
    vec<uint16_t, 8> u16_data{0, 100, 32767, 32768, 40000, 50000, 60000, 65535};
    vec<int16_t, 8> s16_result = convert_to_signed(u16_data);

    EXPECT_EQ(s16_result[0], 0);
    EXPECT_EQ(s16_result[1], 100);
    EXPECT_EQ(s16_result[2], 32767);   // Max positive int16
    EXPECT_EQ(s16_result[3], -32768);  // Wraps to min negative
    EXPECT_LT(s16_result[4], 0);       // Negative
    EXPECT_LT(s16_result[5], 0);       // Negative
    EXPECT_LT(s16_result[6], 0);       // Negative
    EXPECT_EQ(s16_result[7], -1);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s16_result.is_simd_optimized);
    #endif
}

TEST(SignConversionTest, UInt8ToInt8_SameWidth) {
    vec<uint8_t, 16> u8_data{0, 50, 100, 127, 128, 150, 200, 255,
                              10, 20, 30, 40, 129, 180, 220, 254};
    vec<int8_t, 16> s8_result = convert_to_signed(u8_data);

    EXPECT_EQ(s8_result[0], 0);
    EXPECT_EQ(s8_result[1], 50);
    EXPECT_EQ(s8_result[2], 100);
    EXPECT_EQ(s8_result[3], 127);   // Max positive int8
    EXPECT_EQ(s8_result[4], -128);  // Wraps to min negative
    EXPECT_LT(s8_result[5], 0);     // Negative
    EXPECT_LT(s8_result[6], 0);     // Negative
    EXPECT_EQ(s8_result[7], -1);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s8_result.is_simd_optimized);
    #endif
}

//-----------------------------------------------------------------------------
// Test 2: Saturating narrowing conversions (uint16 → int8)
//-----------------------------------------------------------------------------

TEST(SignConversionTest, UInt16ToInt8_Saturating_NoOverflow) {
    vec<uint16_t, 8> u16_data{0, 10, 50, 100, 120, 127, 50, 30};
    vec<int8_t, 8> s8_result = convert_to_signed_sat(u16_data);

    // All values within [0, 127] range
    EXPECT_EQ(s8_result[0], 0);
    EXPECT_EQ(s8_result[1], 10);
    EXPECT_EQ(s8_result[2], 50);
    EXPECT_EQ(s8_result[3], 100);
    EXPECT_EQ(s8_result[4], 120);
    EXPECT_EQ(s8_result[5], 127);
    EXPECT_EQ(s8_result[6], 50);
    EXPECT_EQ(s8_result[7], 30);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s8_result.is_simd_optimized);
    #endif
}

TEST(SignConversionTest, UInt16ToInt8_Saturating_WithOverflow) {
    vec<uint16_t, 8> u16_data{0, 127, 128, 200, 300, 1000, 65535, 50};
    vec<int8_t, 8> s8_result = convert_to_signed_sat(u16_data);

    // Values > 127 saturate to 127
    EXPECT_EQ(s8_result[0], 0);
    EXPECT_EQ(s8_result[1], 127);   // Within range
    EXPECT_EQ(s8_result[2], 127);   // Saturated
    EXPECT_EQ(s8_result[3], 127);   // Saturated
    EXPECT_EQ(s8_result[4], 127);   // Saturated
    EXPECT_EQ(s8_result[5], 127);   // Saturated
    EXPECT_EQ(s8_result[6], 127);   // Saturated
    EXPECT_EQ(s8_result[7], 50);    // Within range

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s8_result.is_simd_optimized);
    #endif
}

TEST(SignConversionTest, UInt16ToInt8_Saturating_AllSaturate) {
    vec<uint16_t, 8> u16_data{128, 200, 300, 500, 1000, 5000, 10000, 65535};
    vec<int8_t, 8> s8_result = convert_to_signed_sat(u16_data);

    // All values saturate to 127
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(s8_result[i], 127) << "Index " << i;
    }

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s8_result.is_simd_optimized);
    #endif
}

//-----------------------------------------------------------------------------
// Test 3: Real-world scenario - Image processing
//-----------------------------------------------------------------------------

TEST(SignConversionTest, ImageProcessing_Scenario) {
    // Simulate unsigned pixel values that need signed representation
    vec<uint16_t, 8> pixel_diff{0, 50, 100, 127, 200, 300, 500, 1000};

    // Convert to signed with saturation for difference calculation
    vec<int8_t, 8> signed_diff = convert_to_signed_sat(pixel_diff);

    // Check saturation behavior
    EXPECT_EQ(signed_diff[0], 0);
    EXPECT_EQ(signed_diff[1], 50);
    EXPECT_EQ(signed_diff[2], 100);
    EXPECT_EQ(signed_diff[3], 127);
    EXPECT_EQ(signed_diff[4], 127);  // Saturated
    EXPECT_EQ(signed_diff[5], 127);  // Saturated
    EXPECT_EQ(signed_diff[6], 127);  // Saturated
    EXPECT_EQ(signed_diff[7], 127);  // Saturated
}

//-----------------------------------------------------------------------------
// Test 4: Narrow register types (64-bit)
//-----------------------------------------------------------------------------

TEST(SignConversionTest, UInt8x8ToInt8x8_SameWidth) {
    vec<uint8_t, 8> u8_data{0, 50, 100, 127, 128, 200, 250, 255};
    vec<int8_t, 8> s8_result = convert_to_signed(u8_data);

    EXPECT_EQ(s8_result[0], 0);
    EXPECT_EQ(s8_result[1], 50);
    EXPECT_EQ(s8_result[2], 100);
    EXPECT_EQ(s8_result[3], 127);
    EXPECT_EQ(s8_result[4], -128);  // Wraps
    EXPECT_LT(s8_result[5], 0);     // Negative
    EXPECT_LT(s8_result[6], 0);     // Negative
    EXPECT_EQ(s8_result[7], -1);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s8_result.is_simd_optimized);
    #endif
}

TEST(SignConversionTest, UInt16x4ToInt16x4_SameWidth) {
    vec<uint16_t, 4> u16_data{0, 32767, 32768, 65535};
    vec<int16_t, 4> s16_result = convert_to_signed(u16_data);

    EXPECT_EQ(s16_result[0], 0);
    EXPECT_EQ(s16_result[1], 32767);
    EXPECT_EQ(s16_result[2], -32768);
    EXPECT_EQ(s16_result[3], -1);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s16_result.is_simd_optimized);
    #endif
}

TEST(SignConversionTest, UInt32x2ToInt32x2_SameWidth) {
    vec<uint32_t, 2> u32_data{2147483647u, 4294967295u};
    vec<int32_t, 2> s32_result = convert_to_signed(u32_data);

    EXPECT_EQ(s32_result[0], 2147483647);
    EXPECT_EQ(s32_result[1], -1);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(s32_result.is_simd_optimized);
    #endif
}

//-----------------------------------------------------------------------------
// Test 5: Round-trip with widening
//-----------------------------------------------------------------------------

TEST(SignConversionTest, RoundTrip_Widen_Then_ConvertSign) {
    // Start with uint8, widen to uint16, then convert to signed
    vec<uint8_t, 8> u8_original{0, 50, 100, 127, 150, 200, 250, 255};

    // Widen to uint16
    vec<uint16_t, 8> u16_wide = convert_widen(u8_original);

    // Convert to signed (same width as widened)
    vec<int16_t, 8> s16_result = convert_to_signed(u16_wide);

    // All values should be positive (no wrapping)
    EXPECT_EQ(s16_result[0], 0);
    EXPECT_EQ(s16_result[1], 50);
    EXPECT_EQ(s16_result[2], 100);
    EXPECT_EQ(s16_result[3], 127);
    EXPECT_EQ(s16_result[4], 150);
    EXPECT_EQ(s16_result[5], 200);
    EXPECT_EQ(s16_result[6], 250);
    EXPECT_EQ(s16_result[7], 255);
}

//-----------------------------------------------------------------------------
// Test 6: NEON optimization verification
//-----------------------------------------------------------------------------

TEST(SignConversionTest, VerifyNEONOptimization) {
    #ifdef TINY_SIMD_ARM_NEON
    // Verify same-width conversions use NEON
    vec<uint32_t, 4> u32_data{1, 2, 3, 4};
    vec<int32_t, 4> s32_result = convert_to_signed(u32_data);
    EXPECT_TRUE(s32_result.is_simd_optimized);

    vec<uint16_t, 8> u16_data{1, 2, 3, 4, 5, 6, 7, 8};
    vec<int16_t, 8> s16_result = convert_to_signed(u16_data);
    EXPECT_TRUE(s16_result.is_simd_optimized);

    vec<uint8_t, 16> u8_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    vec<int8_t, 16> s8_result = convert_to_signed(u8_data);
    EXPECT_TRUE(s8_result.is_simd_optimized);

    // Verify saturating narrowing uses NEON
    vec<uint16_t, 8> u16_sat{100, 200, 300, 400, 500, 600, 700, 800};
    vec<int8_t, 8> s8_sat = convert_to_signed_sat(u16_sat);
    EXPECT_TRUE(s8_sat.is_simd_optimized);
    #else
    GTEST_SKIP() << "NEON not available on this platform";
    #endif
}
