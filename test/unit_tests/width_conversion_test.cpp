#include <gtest/gtest.h>
#include "core/tiny_simd.hpp"

using namespace tiny_simd;

//=============================================================================
// Phase 2 Width Conversion Tests
//=============================================================================

//-----------------------------------------------------------------------------
// Test 1: Widening conversions (int8 → int16)
//-----------------------------------------------------------------------------

TEST(WidthConversionTest, Widen_Int8_To_Int16) {
    vec<int8_t, 8> data{10, 20, 30, 40, -10, -20, -30, -40};
    vec<int16_t, 8> wide = convert_widen(data);

    EXPECT_EQ(wide[0], 10);
    EXPECT_EQ(wide[1], 20);
    EXPECT_EQ(wide[2], 30);
    EXPECT_EQ(wide[3], 40);
    EXPECT_EQ(wide[4], -10);
    EXPECT_EQ(wide[5], -20);
    EXPECT_EQ(wide[6], -30);
    EXPECT_EQ(wide[7], -40);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(wide.is_simd_optimized);
    #endif
}

TEST(WidthConversionTest, Widen_UInt8_To_UInt16) {
    vec<uint8_t, 8> data{100, 150, 200, 250, 10, 20, 30, 40};
    vec<uint16_t, 8> wide = convert_widen(data);

    EXPECT_EQ(wide[0], 100);
    EXPECT_EQ(wide[1], 150);
    EXPECT_EQ(wide[2], 200);
    EXPECT_EQ(wide[3], 250);
    EXPECT_EQ(wide[4], 10);
    EXPECT_EQ(wide[5], 20);
    EXPECT_EQ(wide[6], 30);
    EXPECT_EQ(wide[7], 40);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(wide.is_simd_optimized);
    #endif
}

//-----------------------------------------------------------------------------
// Test 2: Narrowing conversions (int16 → int8)
//-----------------------------------------------------------------------------

TEST(WidthConversionTest, Narrow_Int16_To_Int8_NoOverflow) {
    vec<int16_t, 8> data{10, 20, 30, 40, -10, -20, -30, -40};
    vec<int8_t, 8> narrow = convert_narrow(data);

    EXPECT_EQ(narrow[0], 10);
    EXPECT_EQ(narrow[1], 20);
    EXPECT_EQ(narrow[2], 30);
    EXPECT_EQ(narrow[3], 40);
    EXPECT_EQ(narrow[4], -10);
    EXPECT_EQ(narrow[5], -20);
    EXPECT_EQ(narrow[6], -30);
    EXPECT_EQ(narrow[7], -40);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(narrow.is_simd_optimized);
    #endif
}

TEST(WidthConversionTest, Narrow_UInt16_To_UInt8_NoOverflow) {
    vec<uint16_t, 8> data{100, 150, 200, 250, 10, 20, 30, 40};
    vec<uint8_t, 8> narrow = convert_narrow(data);

    EXPECT_EQ(narrow[0], 100);
    EXPECT_EQ(narrow[1], 150);
    EXPECT_EQ(narrow[2], 200);
    EXPECT_EQ(narrow[3], 250);
    EXPECT_EQ(narrow[4], 10);
    EXPECT_EQ(narrow[5], 20);
    EXPECT_EQ(narrow[6], 30);
    EXPECT_EQ(narrow[7], 40);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(narrow.is_simd_optimized);
    #endif
}

//-----------------------------------------------------------------------------
// Test 3: Saturating narrowing (prevents overflow)
//-----------------------------------------------------------------------------

TEST(WidthConversionTest, NarrowSat_Int16_To_Int8_WithOverflow) {
    vec<int16_t, 8> data{1000, -1000, 127, -128, 200, -200, 50, -50};
    vec<int8_t, 8> narrow_sat = convert_narrow_sat(data);

    EXPECT_EQ(narrow_sat[0], 127);   // Saturated to max
    EXPECT_EQ(narrow_sat[1], -128);  // Saturated to min
    EXPECT_EQ(narrow_sat[2], 127);   // Within range
    EXPECT_EQ(narrow_sat[3], -128);  // Within range
    EXPECT_EQ(narrow_sat[4], 127);   // Saturated to max
    EXPECT_EQ(narrow_sat[5], -128);  // Saturated to min
    EXPECT_EQ(narrow_sat[6], 50);    // Within range
    EXPECT_EQ(narrow_sat[7], -50);   // Within range

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(narrow_sat.is_simd_optimized);
    #endif
}

TEST(WidthConversionTest, NarrowSat_UInt16_To_UInt8_WithOverflow) {
    vec<uint16_t, 8> data{1000, 255, 200, 300, 500, 100, 0, 50};
    vec<uint8_t, 8> narrow_sat = convert_narrow_sat(data);

    EXPECT_EQ(narrow_sat[0], 255);  // Saturated to max
    EXPECT_EQ(narrow_sat[1], 255);  // Within range
    EXPECT_EQ(narrow_sat[2], 200);  // Within range
    EXPECT_EQ(narrow_sat[3], 255);  // Saturated to max
    EXPECT_EQ(narrow_sat[4], 255);  // Saturated to max
    EXPECT_EQ(narrow_sat[5], 100);  // Within range
    EXPECT_EQ(narrow_sat[6], 0);    // Within range
    EXPECT_EQ(narrow_sat[7], 50);   // Within range

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(narrow_sat.is_simd_optimized);
    #endif
}

//-----------------------------------------------------------------------------
// Test 4: Round-trip conversions
//-----------------------------------------------------------------------------

TEST(WidthConversionTest, RoundTrip_Int8_Int16_Int8) {
    vec<int8_t, 8> original{10, 20, 30, 40, -10, -20, -30, -40};

    // Widen then narrow
    vec<int16_t, 8> wide = convert_widen(original);
    vec<int8_t, 8> result = convert_narrow(wide);

    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[1], 20);
    EXPECT_EQ(result[2], 30);
    EXPECT_EQ(result[3], 40);
    EXPECT_EQ(result[4], -10);
    EXPECT_EQ(result[5], -20);
    EXPECT_EQ(result[6], -30);
    EXPECT_EQ(result[7], -40);
}

//-----------------------------------------------------------------------------
// Test 5: Real-world scenario: Image processing with saturation
//-----------------------------------------------------------------------------

TEST(WidthConversionTest, ImageProcessing_Scenario) {
    // Simulate pixel values
    vec<uint8_t, 8> pixels1{100, 110, 120, 130, 140, 150, 160, 170};
    vec<uint8_t, 8> pixels2{50, 60, 70, 80, 90, 100, 110, 120};

    // Widen to prevent overflow during addition
    vec<uint16_t, 8> wide1 = convert_widen(pixels1);
    vec<uint16_t, 8> wide2 = convert_widen(pixels2);

    // Add
    vec<uint16_t, 8> sum = wide1 + wide2;

    // Narrow back with saturation
    vec<uint8_t, 8> result = convert_narrow_sat(sum);

    EXPECT_EQ(result[0], 150);
    EXPECT_EQ(result[1], 170);
    EXPECT_EQ(result[2], 190);
    EXPECT_EQ(result[3], 210);
    EXPECT_EQ(result[4], 230);
    EXPECT_EQ(result[5], 250);
    EXPECT_EQ(result[6], 255);  // Saturated
    EXPECT_EQ(result[7], 255);  // Saturated
}

//-----------------------------------------------------------------------------
// Test 6: NEON optimization verification
//-----------------------------------------------------------------------------

TEST(WidthConversionTest, VerifyNEONOptimization) {
    #ifdef TINY_SIMD_ARM_NEON
    vec<uint8_t, 8> u8_data{1, 2, 3, 4, 5, 6, 7, 8};
    vec<uint16_t, 8> u16_wide = convert_widen(u8_data);
    EXPECT_TRUE(u16_wide.is_simd_optimized);

    vec<int8_t, 8> s8_data{1, 2, 3, 4, 5, 6, 7, 8};
    vec<int16_t, 8> s16_wide = convert_widen(s8_data);
    EXPECT_TRUE(s16_wide.is_simd_optimized);

    vec<uint16_t, 8> u16_data{100, 200, 300, 400, 500, 600, 700, 800};
    vec<uint8_t, 8> u8_narrow = convert_narrow_sat(u16_data);
    EXPECT_TRUE(u8_narrow.is_simd_optimized);
    #else
    GTEST_SKIP() << "NEON not available on this platform";
    #endif
}
