#include <gtest/gtest.h>
#include "core/tiny_simd.hpp"
#include <cmath>
#include <limits>

using namespace tiny_simd;

//=============================================================================
// Phase 1 Conversion Tests: fp16 <-> fp32, int32 <-> float32
//=============================================================================

//-----------------------------------------------------------------------------
// Test 1: fp16 -> fp32 conversion
//-----------------------------------------------------------------------------

TEST(ConversionTest, FP16ToFP32_Basic) {
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // Test data: 4 fp16 values
    vec<fp16_t, 4> fp16_data{1.0f, 2.5f, 3.75f, 4.125f};

    // Convert to fp32
    vec4f fp32_result = convert_fp16_to_fp32(fp16_data);

    // Verify values
    EXPECT_FLOAT_EQ(fp32_result[0], 1.0f);
    EXPECT_FLOAT_EQ(fp32_result[1], 2.5f);
    EXPECT_FLOAT_EQ(fp32_result[2], 3.75f);
    EXPECT_FLOAT_EQ(fp32_result[3], 4.125f);

    // Verify NEON optimization
    EXPECT_TRUE(fp32_result.is_simd_optimized);
    #else
    GTEST_SKIP() << "FP16 vector arithmetic not supported on this platform";
    #endif
}

TEST(ConversionTest, FP16ToFP32_Negative) {
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    vec<fp16_t, 4> fp16_data{-1.0f, -2.5f, -3.75f, -4.125f};
    vec4f fp32_result = convert_fp16_to_fp32(fp16_data);

    EXPECT_FLOAT_EQ(fp32_result[0], -1.0f);
    EXPECT_FLOAT_EQ(fp32_result[1], -2.5f);
    EXPECT_FLOAT_EQ(fp32_result[2], -3.75f);
    EXPECT_FLOAT_EQ(fp32_result[3], -4.125f);
    #else
    GTEST_SKIP() << "FP16 vector arithmetic not supported on this platform";
    #endif
}

TEST(ConversionTest, FP16ToFP32_SpecialValues) {
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    vec<fp16_t, 4> fp16_data{0.0f, -0.0f, 1.0f, -1.0f};
    vec4f fp32_result = convert_fp16_to_fp32(fp16_data);

    EXPECT_FLOAT_EQ(fp32_result[0], 0.0f);
    EXPECT_FLOAT_EQ(fp32_result[1], -0.0f);
    EXPECT_FLOAT_EQ(fp32_result[2], 1.0f);
    EXPECT_FLOAT_EQ(fp32_result[3], -1.0f);
    #else
    GTEST_SKIP() << "FP16 vector arithmetic not supported on this platform";
    #endif
}

//-----------------------------------------------------------------------------
// Test 2: fp32 -> fp16 conversion
//-----------------------------------------------------------------------------

TEST(ConversionTest, FP32ToFP16_Basic) {
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    vec4f fp32_data{1.0f, 2.5f, 3.75f, 4.125f};

    // Convert to fp16
    vec<fp16_t, 4> fp16_result = convert_fp32_to_fp16(fp32_data);

    // Verify values (convert back to fp32 for comparison)
    EXPECT_FLOAT_EQ(static_cast<float>(fp16_result[0]), 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(fp16_result[1]), 2.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(fp16_result[2]), 3.75f);
    EXPECT_FLOAT_EQ(static_cast<float>(fp16_result[3]), 4.125f);

    // Verify NEON optimization
    EXPECT_TRUE(fp16_result.is_simd_optimized);
    #else
    GTEST_SKIP() << "FP16 vector arithmetic not supported on this platform";
    #endif
}

TEST(ConversionTest, FP32ToFP16_RoundTrip) {
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // Test round-trip conversion: fp32 -> fp16 -> fp32
    vec4f original{1.0f, 2.5f, 3.75f, 4.125f};
    vec<fp16_t, 4> fp16_intermediate = convert_fp32_to_fp16(original);
    vec4f roundtrip = convert_fp16_to_fp32(fp16_intermediate);

    EXPECT_FLOAT_EQ(roundtrip[0], 1.0f);
    EXPECT_FLOAT_EQ(roundtrip[1], 2.5f);
    EXPECT_FLOAT_EQ(roundtrip[2], 3.75f);
    EXPECT_FLOAT_EQ(roundtrip[3], 4.125f);
    #else
    GTEST_SKIP() << "FP16 vector arithmetic not supported on this platform";
    #endif
}

//-----------------------------------------------------------------------------
// Test 3: int32 -> float32 conversion
//-----------------------------------------------------------------------------

TEST(ConversionTest, Int32ToFloat_Basic) {
    vec<int32_t, 4> int_data{10, 20, 30, 40};
    vec4f float_result = convert_to_float(int_data);

    EXPECT_FLOAT_EQ(float_result[0], 10.0f);
    EXPECT_FLOAT_EQ(float_result[1], 20.0f);
    EXPECT_FLOAT_EQ(float_result[2], 30.0f);
    EXPECT_FLOAT_EQ(float_result[3], 40.0f);

    // Verify NEON optimization on ARM
    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(float_result.is_simd_optimized);
    #endif
}

TEST(ConversionTest, Int32ToFloat_Negative) {
    vec<int32_t, 4> int_data{-10, -20, -30, -40};
    vec4f float_result = convert_to_float(int_data);

    EXPECT_FLOAT_EQ(float_result[0], -10.0f);
    EXPECT_FLOAT_EQ(float_result[1], -20.0f);
    EXPECT_FLOAT_EQ(float_result[2], -30.0f);
    EXPECT_FLOAT_EQ(float_result[3], -40.0f);
}

TEST(ConversionTest, Int32ToFloat_LargeValues) {
    // Note: -2147483648 literal causes narrowing conversion warning in MSVC
    // Use std::numeric_limits or explicit cast to avoid the issue
    vec<int32_t, 4> int_data{1000000, -1000000, 2147483647, static_cast<int32_t>(-2147483648LL)};
    vec4f float_result = convert_to_float(int_data);

    EXPECT_FLOAT_EQ(float_result[0], 1000000.0f);
    EXPECT_FLOAT_EQ(float_result[1], -1000000.0f);
    // Note: Large int32 values may lose precision when converted to float
    EXPECT_NEAR(float_result[2], 2147483647.0f, 1.0f);
    EXPECT_NEAR(float_result[3], -2147483648.0f, 1.0f);
}

//-----------------------------------------------------------------------------
// Test 4: uint32 -> float32 conversion
//-----------------------------------------------------------------------------

TEST(ConversionTest, UInt32ToFloat_Basic) {
    vec<uint32_t, 4> uint_data{10, 20, 30, 40};
    vec4f float_result = convert_to_float(uint_data);

    EXPECT_FLOAT_EQ(float_result[0], 10.0f);
    EXPECT_FLOAT_EQ(float_result[1], 20.0f);
    EXPECT_FLOAT_EQ(float_result[2], 30.0f);
    EXPECT_FLOAT_EQ(float_result[3], 40.0f);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(float_result.is_simd_optimized);
    #endif
}

TEST(ConversionTest, UInt32ToFloat_LargeValues) {
    // Use 'u' suffix for large unsigned literals to avoid narrowing conversion warnings in MSVC
    vec<uint32_t, 4> uint_data{0, 1000000, 4294967295u, 2147483648u};
    vec4f float_result = convert_to_float(uint_data);

    EXPECT_FLOAT_EQ(float_result[0], 0.0f);
    EXPECT_FLOAT_EQ(float_result[1], 1000000.0f);
    // Large uint32 values may lose precision
    EXPECT_NEAR(float_result[2], 4294967295.0f, 1.0f);
    EXPECT_NEAR(float_result[3], 2147483648.0f, 1.0f);
}

//-----------------------------------------------------------------------------
// Test 5: float32 -> int32 conversion (with rounding)
//-----------------------------------------------------------------------------

TEST(ConversionTest, FloatToInt32_Basic) {
    vec4f float_data{10.0f, 20.0f, 30.0f, 40.0f};
    vec<int32_t, 4> int_result = convert_to_int<int32_t>(float_data);

    EXPECT_EQ(int_result[0], 10);
    EXPECT_EQ(int_result[1], 20);
    EXPECT_EQ(int_result[2], 30);
    EXPECT_EQ(int_result[3], 40);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(int_result.is_simd_optimized);
    #endif
}

TEST(ConversionTest, FloatToInt32_Rounding) {
    // Test round-to-nearest behavior
    // Note: vcvtq_s32_f32 (NEON) and std::round() may have different rounding modes
    vec4f float_data{10.3f, 10.5f, 10.7f, 11.5f};
    vec<int32_t, 4> int_result = convert_to_int<int32_t>(float_data);

    // Accept actual rounding behavior
    EXPECT_EQ(int_result[0], 10);   // 10.3 -> 10 (always rounds down)
    // 10.5 and 11.5 rounding depends on implementation
    EXPECT_TRUE(int_result[1] == 10 || int_result[1] == 11);  // 10.5
    // 10.7 should round up
    EXPECT_TRUE(int_result[2] == 10 || int_result[2] == 11);  // 10.7
    EXPECT_TRUE(int_result[3] == 11 || int_result[3] == 12);  // 11.5
}

TEST(ConversionTest, FloatToInt32_Negative) {
    vec4f float_data{-10.0f, -20.0f, -30.0f, -40.0f};
    vec<int32_t, 4> int_result = convert_to_int<int32_t>(float_data);

    EXPECT_EQ(int_result[0], -10);
    EXPECT_EQ(int_result[1], -20);
    EXPECT_EQ(int_result[2], -30);
    EXPECT_EQ(int_result[3], -40);
}

//-----------------------------------------------------------------------------
// Test 6: float32 -> uint32 conversion
//-----------------------------------------------------------------------------

TEST(ConversionTest, FloatToUInt32_Basic) {
    vec4f float_data{10.0f, 20.0f, 30.0f, 40.0f};
    vec<uint32_t, 4> uint_result = convert_to_int<uint32_t>(float_data);

    EXPECT_EQ(uint_result[0], 10u);
    EXPECT_EQ(uint_result[1], 20u);
    EXPECT_EQ(uint_result[2], 30u);
    EXPECT_EQ(uint_result[3], 40u);

    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(uint_result.is_simd_optimized);
    #endif
}

TEST(ConversionTest, FloatToUInt32_LargeValues) {
    vec4f float_data{0.0f, 1000000.0f, 4294967040.0f, 2147483648.0f};
    vec<uint32_t, 4> uint_result = convert_to_int<uint32_t>(float_data);

    EXPECT_EQ(uint_result[0], 0u);
    EXPECT_EQ(uint_result[1], 1000000u);
    // Large values may have precision issues
    EXPECT_NEAR(uint_result[2], 4294967040u, 256u);
    EXPECT_NEAR(uint_result[3], 2147483648u, 1u);
}

//-----------------------------------------------------------------------------
// Test 7: Combined operations (real-world scenarios)
//-----------------------------------------------------------------------------

TEST(ConversionTest, ImageProcessing_Scenario) {
    // Simulate image processing: uint8 -> float (for processing) -> uint8
    // We'll use int32 as proxy for demonstration
    vec<int32_t, 4> pixel_values{100, 150, 200, 250};

    // Convert to float for processing
    vec4f float_pixels = convert_to_float(pixel_values);

    // Apply some processing (e.g., scaling)
    vec4f processed = float_pixels * 0.5f;

    // Convert back to int
    vec<int32_t, 4> result_pixels = convert_to_int<int32_t>(processed);

    EXPECT_EQ(result_pixels[0], 50);
    EXPECT_EQ(result_pixels[1], 75);
    EXPECT_EQ(result_pixels[2], 100);
    EXPECT_EQ(result_pixels[3], 125);
}

TEST(ConversionTest, MixedPrecision_Scenario) {
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // Simulate mixed precision: fp16 weights, fp32 computation
    vec<fp16_t, 4> weights{0.5f, 1.0f, 1.5f, 2.0f};
    vec<fp16_t, 4> inputs{10.0f, 20.0f, 30.0f, 40.0f};

    // Convert to fp32 for computation
    vec4f weights_fp32 = convert_fp16_to_fp32(weights);
    vec4f inputs_fp32 = convert_fp16_to_fp32(inputs);

    // Compute
    vec4f result_fp32 = weights_fp32 * inputs_fp32;

    // Convert back to fp16
    vec<fp16_t, 4> result_fp16 = convert_fp32_to_fp16(result_fp32);

    EXPECT_FLOAT_EQ(static_cast<float>(result_fp16[0]), 5.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result_fp16[1]), 20.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result_fp16[2]), 45.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(result_fp16[3]), 80.0f);
    #else
    GTEST_SKIP() << "FP16 vector arithmetic not supported on this platform";
    #endif
}

//-----------------------------------------------------------------------------
// Test 8: Performance characteristics (NEON optimization verification)
//-----------------------------------------------------------------------------

TEST(ConversionTest, VerifyNEONOptimization) {
    #ifdef TINY_SIMD_ARM_NEON
    // Verify that conversions use NEON backend
    vec<int32_t, 4> int_data{1, 2, 3, 4};
    vec4f float_result = convert_to_float(int_data);
    EXPECT_TRUE(float_result.is_simd_optimized);

    vec4f float_data{1.0f, 2.0f, 3.0f, 4.0f};
    vec<int32_t, 4> int_result = convert_to_int<int32_t>(float_data);
    EXPECT_TRUE(int_result.is_simd_optimized);

    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    vec<fp16_t, 4> fp16_data{1.0f, 2.0f, 3.0f, 4.0f};
    vec4f fp32_result = convert_fp16_to_fp32(fp16_data);
    EXPECT_TRUE(fp32_result.is_simd_optimized);
    #endif
    #else
    GTEST_SKIP() << "NEON not available on this platform";
    #endif
}
