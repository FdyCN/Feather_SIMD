/**
 * @file neon_datatypes_test.cpp
 * @brief Comprehensive test for all NEON-supported data types
 *
 * Tests all 8 data types required:
 * - uint8_t (16 elements)
 * - int8_t (16 elements)
 * - uint16_t (8 elements)
 * - int16_t (8 elements)
 * - uint32_t (4 elements)
 * - int32_t (4 elements)
 * - float16_t/fp16_t (8 elements)
 * - float32_t/float (2 and 4 elements)
 */

#include "core/tiny_simd.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace tiny_simd;

//=============================================================================
// Helper Functions
//=============================================================================

template<typename T>
inline bool values_equal(T a, T b, T tolerance = T(0)) {
    return std::abs(a - b) <= tolerance;
}

//=============================================================================
// Test: NEON Optimization Detection
//=============================================================================

TEST(NEONDataTypesTest, OptimizationDetection) {
    // Verify SIMD optimization is enabled for all required types
#ifdef TINY_SIMD_ARM_NEON
    // float32
    EXPECT_TRUE((vec<float, 2>::is_simd_optimized)) << "vec<float, 2> should use NEON";
    EXPECT_TRUE((vec<float, 4>::is_simd_optimized)) << "vec<float, 4> should use NEON";

    // int32/uint32
    EXPECT_TRUE((vec<int32_t, 4>::is_simd_optimized)) << "vec<int32_t, 4> should use NEON";
    EXPECT_TRUE((vec<uint32_t, 4>::is_simd_optimized)) << "vec<uint32_t, 4> should use NEON";

    // int16/uint16
    EXPECT_TRUE((vec<int16_t, 8>::is_simd_optimized)) << "vec<int16_t, 8> should use NEON";
    EXPECT_TRUE((vec<uint16_t, 8>::is_simd_optimized)) << "vec<uint16_t, 8> should use NEON";

    // int8/uint8
    EXPECT_TRUE((vec<int8_t, 16>::is_simd_optimized)) << "vec<int8_t, 16> should use NEON";
    EXPECT_TRUE((vec<uint8_t, 16>::is_simd_optimized)) << "vec<uint8_t, 16> should use NEON";

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // fp16
    EXPECT_TRUE((vec<fp16_t, 8>::is_simd_optimized)) << "vec<fp16_t, 8> should use NEON";
#endif
#endif
}

//=============================================================================
// Test: uint8_t (16 elements)
//=============================================================================

TEST(NEONDataTypesTest, Uint8Operations) {
    vec<uint8_t, 16> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    vec<uint8_t, 16> b{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    // Addition
    auto sum = a + b;
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(sum[i], 17) << "uint8_t addition failed at index " << i;
    }

    // Subtraction
    auto diff = b - a;
    EXPECT_EQ(diff[0], 15);
    EXPECT_EQ(diff[15], 241);  // 1 - 16 = -15 = 241 (unsigned wraparound)

    // Multiplication
    vec<uint8_t, 16> c{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    auto prod = a * c;
    EXPECT_EQ(prod[0], 2);
    EXPECT_EQ(prod[7], 16);

    // Min/Max
    auto min_val = min(a, b);
    auto max_val = max(a, b);
    EXPECT_EQ(min_val[0], 1);
    EXPECT_EQ(max_val[0], 16);
}

//=============================================================================
// Test: int8_t (16 elements)
//=============================================================================

TEST(NEONDataTypesTest, Int8Operations) {
    vec<int8_t, 16> a{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8};
    vec<int8_t, 16> b{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};

    // Addition
    auto sum = a + b;
    EXPECT_EQ(sum[0], 11);
    EXPECT_EQ(sum[8], 9);  // -1 + 10

    // Subtraction
    auto diff = a - b;
    EXPECT_EQ(diff[0], -9);
    EXPECT_EQ(diff[8], -11);

    // Negation
    auto neg = -a;
    EXPECT_EQ(neg[0], -1);
    EXPECT_EQ(neg[8], 1);

    // Absolute value
    auto abs_val = abs(a);
    EXPECT_EQ(abs_val[0], 1);
    EXPECT_EQ(abs_val[8], 1);
    EXPECT_EQ(abs_val[15], 8);
}

//=============================================================================
// Test: uint16_t (8 elements)
//=============================================================================

TEST(NEONDataTypesTest, Uint16Operations) {
    vec<uint16_t, 8> a{100, 200, 300, 400, 500, 600, 700, 800};
    vec<uint16_t, 8> b{10, 20, 30, 40, 50, 60, 70, 80};

    // Addition
    auto sum = a + b;
    EXPECT_EQ(sum[0], 110);
    EXPECT_EQ(sum[7], 880);

    // Subtraction
    auto diff = a - b;
    EXPECT_EQ(diff[0], 90);
    EXPECT_EQ(diff[7], 720);

    // Multiplication
    vec<uint16_t, 8> c{2, 2, 2, 2, 2, 2, 2, 2};
    auto prod = b * c;
    EXPECT_EQ(prod[0], 20);
    EXPECT_EQ(prod[7], 160);

    // Min/Max
    auto min_val = min(a, b);
    auto max_val = max(a, b);
    EXPECT_EQ(min_val[0], 10);
    EXPECT_EQ(max_val[0], 100);
}

//=============================================================================
// Test: int16_t (8 elements) - NEWLY ADDED
//=============================================================================

TEST(NEONDataTypesTest, Int16Operations) {
    vec<int16_t, 8> a{100, 200, 300, 400, -100, -200, -300, -400};
    vec<int16_t, 8> b{10, 20, 30, 40, 50, 60, 70, 80};

    // Addition
    auto sum = a + b;
    EXPECT_EQ(sum[0], 110);
    EXPECT_EQ(sum[4], -50);  // -100 + 50

    // Subtraction
    auto diff = a - b;
    EXPECT_EQ(diff[0], 90);
    EXPECT_EQ(diff[4], -150);

    // Multiplication
    vec<int16_t, 8> c{2, 2, 2, 2, 2, 2, 2, 2};
    auto prod = a * c;
    EXPECT_EQ(prod[0], 200);
    EXPECT_EQ(prod[4], -200);

    // Negation
    auto neg = -a;
    EXPECT_EQ(neg[0], -100);
    EXPECT_EQ(neg[4], 100);

    // Absolute value
    auto abs_val = abs(a);
    EXPECT_EQ(abs_val[0], 100);
    EXPECT_EQ(abs_val[4], 100);
    EXPECT_EQ(abs_val[7], 400);

    // Min/Max
    auto min_val = min(a, b);
    auto max_val = max(a, b);
    EXPECT_EQ(min_val[4], -100);
    EXPECT_EQ(max_val[0], 100);
}

//=============================================================================
// Test: uint32_t (4 elements)
//=============================================================================

TEST(NEONDataTypesTest, Uint32Operations) {
    vec<uint32_t, 4> a{1000, 2000, 3000, 4000};
    vec<uint32_t, 4> b{100, 200, 300, 400};

    // Addition
    auto sum = a + b;
    EXPECT_EQ(sum[0], 1100);
    EXPECT_EQ(sum[3], 4400);

    // Subtraction
    auto diff = a - b;
    EXPECT_EQ(diff[0], 900);
    EXPECT_EQ(diff[3], 3600);

    // Multiplication
    vec<uint32_t, 4> c{2, 2, 2, 2};
    auto prod = a * c;
    EXPECT_EQ(prod[0], 2000);
    EXPECT_EQ(prod[3], 8000);
}

//=============================================================================
// Test: int32_t (4 elements)
//=============================================================================

TEST(NEONDataTypesTest, Int32Operations) {
    vec<int32_t, 4> a{1000, 2000, -3000, -4000};
    vec<int32_t, 4> b{100, 200, 300, 400};

    // Addition
    auto sum = a + b;
    EXPECT_EQ(sum[0], 1100);
    EXPECT_EQ(sum[2], -2700);

    // Subtraction
    auto diff = a - b;
    EXPECT_EQ(diff[0], 900);
    EXPECT_EQ(diff[2], -3300);

    // Multiplication
    vec<int32_t, 4> c{2, 2, 2, 2};
    auto prod = a * c;
    EXPECT_EQ(prod[0], 2000);
    EXPECT_EQ(prod[2], -6000);

    // Negation
    auto neg = -a;
    EXPECT_EQ(neg[0], -1000);
    EXPECT_EQ(neg[2], 3000);

    // Absolute value
    auto abs_val = abs(a);
    EXPECT_EQ(abs_val[0], 1000);
    EXPECT_EQ(abs_val[2], 3000);
}

//=============================================================================
// Test: float32_t (2 elements)
//=============================================================================

TEST(NEONDataTypesTest, Float32x2Operations) {
    vec<float, 2> a{1.5f, 2.5f};
    vec<float, 2> b{0.5f, 1.5f};

    // Addition
    auto sum = a + b;
    EXPECT_FLOAT_EQ(sum[0], 2.0f);
    EXPECT_FLOAT_EQ(sum[1], 4.0f);

    // Subtraction
    auto diff = a - b;
    EXPECT_FLOAT_EQ(diff[0], 1.0f);
    EXPECT_FLOAT_EQ(diff[1], 1.0f);

    // Multiplication
    auto prod = a * b;
    EXPECT_FLOAT_EQ(prod[0], 0.75f);
    EXPECT_FLOAT_EQ(prod[1], 3.75f);

    // Division
    auto quot = a / b;
    EXPECT_NEAR(quot[0], 3.0f, 1e-3f);
    EXPECT_NEAR(quot[1], 1.666667f, 1e-3f);
}

//=============================================================================
// Test: float32_t (4 elements)
//=============================================================================

TEST(NEONDataTypesTest, Float32x4Operations) {
    vec<float, 4> a{1.0f, 2.0f, 3.0f, 4.0f};
    vec<float, 4> b{5.0f, 6.0f, 7.0f, 8.0f};

    // Addition
    auto sum = a + b;
    EXPECT_FLOAT_EQ(sum[0], 6.0f);
    EXPECT_FLOAT_EQ(sum[3], 12.0f);

    // Subtraction
    auto diff = b - a;
    EXPECT_FLOAT_EQ(diff[0], 4.0f);
    EXPECT_FLOAT_EQ(diff[3], 4.0f);

    // Multiplication
    auto prod = a * b;
    EXPECT_FLOAT_EQ(prod[0], 5.0f);
    EXPECT_FLOAT_EQ(prod[3], 32.0f);

    // Division
    auto quot = b / a;
    EXPECT_NEAR(quot[0], 5.0f, 1e-3f);
    EXPECT_NEAR(quot[3], 2.0f, 1e-3f);

    // Negation
    auto neg = -a;
    EXPECT_FLOAT_EQ(neg[0], -1.0f);
    EXPECT_FLOAT_EQ(neg[3], -4.0f);

    // Absolute value
    vec<float, 4> c{-1.0f, -2.0f, 3.0f, -4.0f};
    auto abs_val = abs(c);
    EXPECT_FLOAT_EQ(abs_val[0], 1.0f);
    EXPECT_FLOAT_EQ(abs_val[1], 2.0f);
    EXPECT_FLOAT_EQ(abs_val[3], 4.0f);
}

//=============================================================================
// Test: fp16_t (8 elements) - Conditional
//=============================================================================

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
TEST(NEONDataTypesTest, Float16Operations) {
    vec<fp16_t, 8> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    vec<fp16_t, 8> b{0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};

    // Addition
    auto sum = a + b;
    EXPECT_NEAR(static_cast<float>(sum[0]), 1.5f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(sum[7]), 12.0f, 1e-2f);

    // Subtraction
    auto diff = a - b;
    EXPECT_NEAR(static_cast<float>(diff[0]), 0.5f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(diff[7]), 4.0f, 1e-2f);

    // Multiplication
    auto prod = a * b;
    EXPECT_NEAR(static_cast<float>(prod[0]), 0.5f, 1e-2f);
    EXPECT_NEAR(static_cast<float>(prod[7]), 32.0f, 1e-1f);
}
#else
TEST(NEONDataTypesTest, Float16NotSupported) {
    GTEST_SKIP() << "FP16 vector arithmetic not supported on this platform";
}
#endif

//=============================================================================
// Test: Data Type Summary
//=============================================================================

TEST(NEONDataTypesTest, DataTypeSummary) {
    std::cout << "\n=== NEON Data Types Support Summary ===" << std::endl;
    std::cout << "uint8_t  x16: " << (vec<uint8_t, 16>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
    std::cout << "int8_t   x16: " << (vec<int8_t, 16>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
    std::cout << "uint16_t x8:  " << (vec<uint16_t, 8>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
    std::cout << "int16_t  x8:  " << (vec<int16_t, 8>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
    std::cout << "uint32_t x4:  " << (vec<uint32_t, 4>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
    std::cout << "int32_t  x4:  " << (vec<int32_t, 4>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
    std::cout << "float    x2:  " << (vec<float, 2>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
    std::cout << "float    x4:  " << (vec<float, 4>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    std::cout << "fp16_t   x8:  " << (vec<fp16_t, 8>::is_simd_optimized ? "✓ NEON" : "✗ Scalar") << std::endl;
#else
    std::cout << "fp16_t   x8:  ✗ Not supported" << std::endl;
#endif
    std::cout << "=======================================" << std::endl;
}
