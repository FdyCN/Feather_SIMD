#include <gtest/gtest.h>
#include "core/tiny_simd.hpp"
#include <cmath>

using namespace tiny_simd;

//=============================================================================
// Bitwise Tests
//=============================================================================

TEST(SimdOpsTest, BitwiseAnd) {
    vec4i a{0xFF, 0x00, 0xAA, 0x55};
    vec4i b{0x0F, 0xFF, 0x55, 0xAA};
    vec4i result = a & b;

    EXPECT_EQ(result[0], 0x0F);
    EXPECT_EQ(result[1], 0x00);
    EXPECT_EQ(result[2], 0x00); // AA & 55 = 10101010 & 01010101 = 0
    EXPECT_EQ(result[3], 0x00);
}

TEST(SimdOpsTest, BitwiseOr) {
    vec4i a{0xF0, 0x00, 0xAA, 0x55};
    vec4i b{0x0F, 0xFF, 0x55, 0xAA};
    vec4i result = a | b;

    EXPECT_EQ(result[0], 0xFF);
    EXPECT_EQ(result[1], 0xFF);
    EXPECT_EQ(result[2], 0xFF); // AA | 55 = 10101010 | 01010101 = 11111111
    EXPECT_EQ(result[3], 0xFF);
}

TEST(SimdOpsTest, BitwiseXor) {
    vec4i a{0xFF, 0x00, 0xAA, 0x55};
    vec4i b{0x0F, 0xFF, 0x55, 0xAA};
    vec4i result = a ^ b;

    EXPECT_EQ(result[0], 0xF0);
    EXPECT_EQ(result[1], 0xFF);
    EXPECT_EQ(result[2], 0xFF); // AA ^ 55 = 11111111
    EXPECT_EQ(result[3], 0xFF);
}

TEST(SimdOpsTest, BitwiseNot) {
    vec4i a{0, -1, 0x55555555, static_cast<int32_t>(0xAAAAAAAA)};
    vec4i result = ~a;

    EXPECT_EQ(result[0], -1);
    EXPECT_EQ(result[1], 0);
    EXPECT_EQ(result[2], static_cast<int32_t>(0xAAAAAAAA));
    EXPECT_EQ(result[3], 0x55555555);
}

TEST(SimdOpsTest, BitwiseAndNot) {
    vec4i a{0xFF, 0xFF, 0xAA, 0x55};
    vec4i b{0x0F, 0xF0, 0x55, 0xAA};
    // a & ~b
    vec4i result = bitwise_andnot(a, b);

    EXPECT_EQ(result[0], 0xF0); // FF & ~0F = FF & F0 = F0
    EXPECT_EQ(result[1], 0x0F); // FF & ~F0 = FF & 0F = 0F
    EXPECT_EQ(result[2], 0xAA); // AA & ~55 = AA & AA = AA
    // EXPECT_EQ(result[3], 0x00); // Removed incorrect expectation
    // Let's recheck logic:
    // a = 0x55 (01010101)
    // b = 0xAA (10101010)
    // ~b = 0x55 (01010101) (assuming 8-bit for simplicity, but pattern holds)
    // a & ~b = 0x55 & 0x55 = 0x55.
    EXPECT_EQ(result[3], 0x55);
}

//=============================================================================
// Shift Tests
//=============================================================================

TEST(SimdOpsTest, ShiftLeft) {
    vec4i a{1, 2, 4, 8};
    vec4i result = a << 2;

    EXPECT_EQ(result[0], 4);
    EXPECT_EQ(result[1], 8);
    EXPECT_EQ(result[2], 16);
    EXPECT_EQ(result[3], 32);
}

TEST(SimdOpsTest, ShiftRightSigned) {
    vec4i a{4, 8, -16, -32};
    vec4i result = a >> 2;

    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], -4); // Arithmetic shift preserves sign
    EXPECT_EQ(result[3], -8);
}

TEST(SimdOpsTest, ShiftRightUnsigned) {
    vec4ui a{4, 8, 0xFFFFFFFF, 0x80000000};
    vec4ui result = a >> 2;

    EXPECT_EQ(result[0], 1u);
    EXPECT_EQ(result[1], 2u);
    EXPECT_EQ(result[2], 0x3FFFFFFFu); // Logical shift fills with zero
    EXPECT_EQ(result[3], 0x20000000u);
}

//=============================================================================
// FMA Tests
//=============================================================================

TEST(SimdOpsTest, FMA) {
    vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
    vec4f b{2.0f, 3.0f, 4.0f, 5.0f};
    vec4f c{10.0f, 20.0f, 30.0f, 40.0f};
    
    // a * b + c
    vec4f result = fma(a, b, c);

    EXPECT_FLOAT_EQ(result[0], 12.0f); // 1*2 + 10 = 12
    EXPECT_FLOAT_EQ(result[1], 26.0f); // 2*3 + 20 = 26
    EXPECT_FLOAT_EQ(result[2], 42.0f); // 3*4 + 30 = 42
    EXPECT_FLOAT_EQ(result[3], 60.0f); // 4*5 + 40 = 60
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
