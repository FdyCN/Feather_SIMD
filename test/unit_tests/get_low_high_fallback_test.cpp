#include <gtest/gtest.h>
#include "core/tiny_simd.hpp"
#include <iostream>

using namespace tiny_simd;

// Test that get_low/get_high work with both NEON and scalar backends
class GetLowHighFallbackTest : public ::testing::Test {};

//=============================================================================
// Test scalar fallback for sizes where NEON has no advantage
//=============================================================================

TEST_F(GetLowHighFallbackTest, ScalarBackendUint8x32) {
    // uint8_t with N=32 should use scalar_backend (NEON max is 16)
    vec<uint8_t, 32, scalar_backend> data;
    for (size_t i = 0; i < 32; ++i) {
        alignas(32) uint8_t temp[32];
        for (size_t j = 0; j < 32; ++j) temp[j] = j;
        data = vec<uint8_t, 32, scalar_backend>(temp);
    }

    // Should use scalar implementation from base.hpp
    auto low = get_low(data);   // uint8_t[16]
    auto high = get_high(data); // uint8_t[16]

    // Verify results
    EXPECT_EQ(low[0], 0);
    EXPECT_EQ(low[15], 15);
    EXPECT_EQ(high[0], 16);
    EXPECT_EQ(high[15], 31);

    // These should NOT be SIMD optimized (scalar backend)
    EXPECT_FALSE(low.is_simd_optimized);
    EXPECT_FALSE(high.is_simd_optimized);

    std::cout << "✓ Scalar fallback works for non-NEON sizes" << std::endl;
}

TEST_F(GetLowHighFallbackTest, NeonBackendUint8x16) {
    // uint8_t with N=16 should use neon_backend
    vec<uint8_t, 16> data;
    alignas(16) uint8_t temp[16];
    for (size_t i = 0; i < 16; ++i) temp[i] = i * 10;
    data = vec<uint8_t, 16>(temp);

    // Should use NEON implementation from neon.hpp
    auto low = get_low(data);   // uint8_t[8]
    auto high = get_high(data); // uint8_t[8]

    // Verify results
    EXPECT_EQ(low[0], 0);
    EXPECT_EQ(low[7], 70);
    EXPECT_EQ(high[0], 80);
    EXPECT_EQ(high[7], 150);

    // These SHOULD be SIMD optimized (neon backend)
    EXPECT_TRUE(data.is_simd_optimized);
    EXPECT_TRUE(low.is_simd_optimized);
    EXPECT_TRUE(high.is_simd_optimized);

    std::cout << "✓ NEON optimization correctly selected for supported sizes" << std::endl;
}

TEST_F(GetLowHighFallbackTest, CompareNeonVsScalar) {
    // Same data, different backends
    alignas(16) uint16_t test_data[8] = {100, 200, 300, 400, 500, 600, 700, 800};

    // NEON backend (auto-selected)
    vec<uint16_t, 8> neon_vec(test_data);
    auto neon_low = get_low(neon_vec);
    auto neon_high = get_high(neon_vec);

    // Scalar backend (explicit)
    vec<uint16_t, 8, scalar_backend> scalar_vec(test_data);
    auto scalar_low = get_low(scalar_vec);
    auto scalar_high = get_high(scalar_vec);

    // Results should be identical
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(neon_low[i], scalar_low[i]);
        EXPECT_EQ(neon_high[i], scalar_high[i]);
    }

    // But NEON should be optimized, scalar should not
    EXPECT_TRUE(neon_vec.is_simd_optimized);
    EXPECT_TRUE(neon_low.is_simd_optimized);
    EXPECT_TRUE(neon_high.is_simd_optimized);

    EXPECT_FALSE(scalar_vec.is_simd_optimized);
    EXPECT_FALSE(scalar_low.is_simd_optimized);
    EXPECT_FALSE(scalar_high.is_simd_optimized);

    std::cout << "✓ NEON and scalar backends produce identical results" << std::endl;
}

//=============================================================================
// Test unified interface behavior
//=============================================================================

TEST_F(GetLowHighFallbackTest, UnifiedInterfaceInt32) {
    vec4i data{10, 20, 30, 40};

    // Unified interface - should auto-select NEON
    auto low = get_low(data);   // int32x2
    auto high = get_high(data); // int32x2

    EXPECT_EQ(low[0], 10);
    EXPECT_EQ(low[1], 20);
    EXPECT_EQ(high[0], 30);
    EXPECT_EQ(high[1], 40);

    // Verify NEON optimization
    EXPECT_TRUE(data.is_simd_optimized);
    EXPECT_TRUE(low.is_simd_optimized);
    EXPECT_TRUE(high.is_simd_optimized);
}

TEST_F(GetLowHighFallbackTest, UnifiedInterfaceFloat) {
    vec4f positions{1.0f, 2.0f, 3.0f, 4.0f};

    // Unified interface - should auto-select NEON
    auto xy = get_low(positions);   // float32x2
    auto zw = get_high(positions);  // float32x2

    EXPECT_FLOAT_EQ(xy[0], 1.0f);
    EXPECT_FLOAT_EQ(xy[1], 2.0f);
    EXPECT_FLOAT_EQ(zw[0], 3.0f);
    EXPECT_FLOAT_EQ(zw[1], 4.0f);

    // Verify NEON optimization
    EXPECT_TRUE(positions.is_simd_optimized);
    EXPECT_TRUE(xy.is_simd_optimized);
    EXPECT_TRUE(zw.is_simd_optimized);
}
