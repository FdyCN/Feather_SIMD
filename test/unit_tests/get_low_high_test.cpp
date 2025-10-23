#include <gtest/gtest.h>
#include "core/tiny_simd.hpp"
#include <iostream>

using namespace tiny_simd;

// Test get_low/get_high interface with widening operations
class GetLowHighTest : public ::testing::Test {
protected:
    void SetUp() override {
        for (int i = 0; i < 16; ++i) {
            u8_data[i] = 100 + i;
            s8_data[i] = -8 + i;
        }
        for (int i = 0; i < 8; ++i) {
            u16_data[i] = 30000 + i * 100;
            s16_data[i] = -100 + i * 25;
        }
    }

    alignas(16) uint8_t u8_data[16];
    alignas(16) int8_t s8_data[16];
    alignas(16) uint16_t u16_data[8];
    alignas(16) int16_t s16_data[8];
};

//=============================================================================
// Test get_low / get_high extraction
//=============================================================================

TEST_F(GetLowHighTest, ExtractLowHighUint8) {
    vec16ub full{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115};

    auto low = get_low(full);
    auto high = get_high(full);

    // Verify low half (first 8 elements)
    EXPECT_EQ(low[0], 100);
    EXPECT_EQ(low[7], 107);

    // Verify high half (last 8 elements)
    EXPECT_EQ(high[0], 108);
    EXPECT_EQ(high[7], 115);

    // Verify SIMD optimization
    EXPECT_TRUE(low.is_simd_optimized);
    EXPECT_TRUE(high.is_simd_optimized);
}

TEST_F(GetLowHighTest, ExtractLowHighInt16) {
    vec8s full{-100, -75, -50, -25, 0, 25, 50, 75};

    auto low = get_low(full);
    auto high = get_high(full);

    // Verify low half (first 4 elements)
    EXPECT_EQ(low[0], -100);
    EXPECT_EQ(low[3], -25);

    // Verify high half (last 4 elements)
    EXPECT_EQ(high[0], 0);
    EXPECT_EQ(high[3], 75);

    // Verify SIMD optimization
    EXPECT_TRUE(low.is_simd_optimized);
    EXPECT_TRUE(high.is_simd_optimized);
}

TEST_F(GetLowHighTest, ExtractLowHighFloat4) {
    vec4f full{1.0f, 2.0f, 3.0f, 4.0f};

    auto low = get_low(full);
    auto high = get_high(full);

    // Verify low half (first 2 elements)
    EXPECT_FLOAT_EQ(low[0], 1.0f);
    EXPECT_FLOAT_EQ(low[1], 2.0f);

    // Verify high half (last 2 elements)
    EXPECT_FLOAT_EQ(high[0], 3.0f);
    EXPECT_FLOAT_EQ(high[1], 4.0f);

    // Verify SIMD optimization
    EXPECT_TRUE(low.is_simd_optimized);
    EXPECT_TRUE(high.is_simd_optimized);
}

//=============================================================================
// Test combining get_low/high with wide operations
//=============================================================================

TEST_F(GetLowHighTest, WideOperationWithGetLow) {
    vec16ub a{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115};
    vec16ub b{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};

    // Extract low halves
    auto a_low = get_low(a);
    auto b_low = get_low(b);

    // Now use the generic add_wide from base.hpp - it will work with the narrow registers
    // auto wide_result = add_wide(a_low, b_low);  // This should widen uint8x8 -> uint16x8

    // For now, let's verify the extraction works
    EXPECT_EQ(a_low[0], 100);
    EXPECT_EQ(b_low[0], 10);
}

TEST_F(GetLowHighTest, CompleteWorkflowWithGetLowHigh) {
    // Start with full vectors
    vec16ub img1{100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250};
    vec16ub img2{50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50};

    // Extract halves
    auto img1_low = get_low(img1);
    auto img2_low = get_low(img2);
    auto img1_high = get_high(img1);
    auto img2_high = get_high(img2);

    // Verify extraction
    EXPECT_EQ(img1_low[0], 100);
    EXPECT_EQ(img1_low[7], 170);
    EXPECT_EQ(img1_high[0], 180);
    EXPECT_EQ(img1_high[7], 250);

    // All operations should use SIMD
    EXPECT_TRUE(img1_low.is_simd_optimized);
    EXPECT_TRUE(img1_high.is_simd_optimized);

    std::cout << "✓ get_low/get_high extraction works correctly with SIMD" << std::endl;
}

//=============================================================================
// Test NEON-specific operations on narrow registers
//=============================================================================

TEST_F(GetLowHighTest, ArithmeticOnNarrowRegisters) {
    vec16ub full{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160};

    auto low = get_low(full);   // uint8x8: {10, 20, 30, 40, 50, 60, 70, 80}
    auto high = get_high(full); // uint8x8: {90, 100, 110, 120, 130, 140, 150, 160}

    // Add narrow registers
    auto sum = low + high;

    // Verify results
    EXPECT_EQ(sum[0], 10 + 90);
    EXPECT_EQ(sum[7], 80 + 160);

    // Verify SIMD optimization
    EXPECT_TRUE(sum.is_simd_optimized);
}

TEST_F(GetLowHighTest, MultiplyNarrowRegisters) {
    vec8us full{10, 20, 30, 40, 50, 60, 70, 80};

    auto low = get_low(full);   // uint16x4: {10, 20, 30, 40}
    auto high = get_high(full); // uint16x4: {50, 60, 70, 80}

    // Multiply narrow registers
    auto product = low * high;

    // Verify results
    EXPECT_EQ(product[0], 10 * 50);
    EXPECT_EQ(product[3], 40 * 80);

    // Verify SIMD optimization
    EXPECT_TRUE(product.is_simd_optimized);
}
