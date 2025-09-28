#include "core/tiny_simd.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <chrono>

using namespace tiny_simd;

// 测试整数运算溢出问题和解决方案
class OverflowTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(OverflowTest, Uint8OverflowDemo) {
    // 演示 uint8_t 溢出问题
    vec16ub a{200, 200, 200, 200, 200, 200, 200, 200,
              200, 200, 200, 200, 200, 200, 200, 200};
    vec16ub b{100, 100, 100, 100, 100, 100, 100, 100,
              100, 100, 100, 100, 100, 100, 100, 100};

    // 常规加法 - 会溢出 (200 + 100 = 300 → 44)
    vec16ub regular_sum = a + b;
    EXPECT_EQ(regular_sum[0], 44);  // 300 % 256 = 44

    std::cout << "Regular addition (with overflow): " << (int)regular_sum[0] << std::endl;
}

TEST_F(OverflowTest, WideningArithmetic) {
    // 加宽运算 - 返回更大的数据类型，避免溢出
    vec16ub a{200, 150, 255, 128, 200, 150, 255, 128,
              200, 150, 255, 128, 200, 150, 255, 128};
    vec16ub b{100, 200, 1, 128, 100, 200, 1, 128,
              100, 200, 1, 128, 100, 200, 1, 128};

    // 加宽加法 - uint8 → uint16
    auto wide_sum = add_wide(a, b);
    EXPECT_EQ(wide_sum[0], 300);  // 正确结果，无溢出
    EXPECT_EQ(wide_sum[1], 350);
    EXPECT_EQ(wide_sum[2], 256);
    EXPECT_EQ(wide_sum[3], 256);

    std::cout << "Widening addition results: " << wide_sum[0] << ", "
              << wide_sum[1] << ", " << wide_sum[2] << std::endl;

    // 加宽乘法 - uint8 → uint16
    vec16ub c{15, 16, 17, 18, 15, 16, 17, 18,
              15, 16, 17, 18, 15, 16, 17, 18};
    vec16ub d{20, 21, 22, 23, 20, 21, 22, 23,
              20, 21, 22, 23, 20, 21, 22, 23};

    auto wide_mul = mul_wide(c, d);
    EXPECT_EQ(wide_mul[0], 300);  // 15 * 20 = 300
    EXPECT_EQ(wide_mul[1], 336);  // 16 * 21 = 336
    EXPECT_EQ(wide_mul[2], 374);  // 17 * 22 = 374
    EXPECT_EQ(wide_mul[3], 414);  // 18 * 23 = 414

    std::cout << "Widening multiplication results: " << wide_mul[0] << ", "
              << wide_mul[1] << ", " << wide_mul[2] << std::endl;
}

TEST_F(OverflowTest, SaturatingArithmetic) {
    // 饱和运算 - 限制在数据类型范围内
    vec16ub a{200, 150, 255, 100, 200, 150, 255, 100,
              200, 150, 255, 100, 200, 150, 255, 100};
    vec16ub b{100, 200, 50, 50, 100, 200, 50, 50,
              100, 200, 50, 50, 100, 200, 50, 50};

    // 饱和加法 - 结果限制在 [0, 255]
    vec16ub sat_sum = add_sat(a, b);
    EXPECT_EQ(sat_sum[0], 255);  // min(200 + 100, 255) = 255
    EXPECT_EQ(sat_sum[1], 255);  // min(150 + 200, 255) = 255
    EXPECT_EQ(sat_sum[2], 255);  // min(255 + 50, 255) = 255
    EXPECT_EQ(sat_sum[3], 150);  // 100 + 50 = 150

    std::cout << "Saturating addition results: " << (int)sat_sum[0] << ", "
              << (int)sat_sum[1] << ", " << (int)sat_sum[2] << ", " << (int)sat_sum[3] << std::endl;

    // 饱和减法 - 结果限制在 [0, 255]
    vec16ub c{50, 200, 100, 0, 50, 200, 100, 0,
              50, 200, 100, 0, 50, 200, 100, 0};
    vec16ub d{100, 150, 50, 10, 100, 150, 50, 10,
              100, 150, 50, 10, 100, 150, 50, 10};

    vec16ub sat_sub = sub_sat(c, d);
    EXPECT_EQ(sat_sub[0], 0);   // max(50 - 100, 0) = 0
    EXPECT_EQ(sat_sub[1], 50);  // 200 - 150 = 50
    EXPECT_EQ(sat_sub[2], 50);  // 100 - 50 = 50
    EXPECT_EQ(sat_sub[3], 0);   // max(0 - 10, 0) = 0

    std::cout << "Saturating subtraction results: " << (int)sat_sub[0] << ", "
              << (int)sat_sub[1] << ", " << (int)sat_sub[2] << ", " << (int)sat_sub[3] << std::endl;
}

TEST_F(OverflowTest, NarrowingSaturation) {
    // 测试从加宽结果饱和转换回原类型
    vec16ub a{200, 150, 255, 100, 200, 150, 255, 100,
              200, 150, 255, 100, 200, 150, 255, 100};
    vec16ub b{100, 200, 50, 50, 100, 200, 50, 50,
              100, 200, 50, 50, 100, 200, 50, 50};

    // 先加宽加法
    auto wide_sum = add_wide(a, b);  // 结果: 300, 350, 305, 150, ...

    // 然后窄化饱和转换回uint8
    auto narrow_result = narrow_sat(wide_sum);
    EXPECT_EQ(narrow_result[0], 255);  // min(300, 255) = 255
    EXPECT_EQ(narrow_result[1], 255);  // min(350, 255) = 255
    EXPECT_EQ(narrow_result[2], 255);  // min(305, 255) = 255
    EXPECT_EQ(narrow_result[3], 150);  // 150 (在范围内)

    std::cout << "Narrowing saturation results: " << (int)narrow_result[0] << ", "
              << (int)narrow_result[1] << ", " << (int)narrow_result[2] << ", " << (int)narrow_result[3] << std::endl;
}

TEST_F(OverflowTest, RegisterPairHandling) {
    // 测试NEON寄存器对处理
    // uint8x16 (1个寄存器) → uint16x16 (2个寄存器)
    vec16ub a{255, 255, 255, 255, 255, 255, 255, 255,
              255, 255, 255, 255, 255, 255, 255, 255};
    vec16ub b{1, 2, 3, 4, 5, 6, 7, 8,
              9, 10, 11, 12, 13, 14, 15, 16};

    // 加宽加法应该正确处理所有16个元素
    auto wide_result = add_wide(a, b);

    EXPECT_EQ(wide_result.size(), 16);  // 确保有16个元素
    EXPECT_EQ(wide_result[0], 256);     // 255 + 1 = 256
    EXPECT_EQ(wide_result[7], 263);     // 255 + 8 = 263
    EXPECT_EQ(wide_result[8], 264);     // 255 + 9 = 264 (高8字节)
    EXPECT_EQ(wide_result[15], 271);    // 255 + 16 = 271

    std::cout << "Register pair handling - element 0: " << wide_result[0]
              << ", element 8: " << wide_result[8]
              << ", element 15: " << wide_result[15] << std::endl;
}

TEST_F(OverflowTest, Uint16Operations) {
    // uint16 类型的加宽运算
    vec8us a{30000, 40000, 50000, 60000, 30000, 40000, 50000, 60000};
    vec8us b{20000, 30000, 20000, 10000, 20000, 30000, 20000, 10000};

    // 加宽加法 - uint16 → uint32
    auto wide_sum = add_wide(a, b);
    EXPECT_EQ(wide_sum[0], 50000);   // 30000 + 20000 = 50000
    EXPECT_EQ(wide_sum[1], 70000);   // 40000 + 30000 = 70000
    EXPECT_EQ(wide_sum[2], 70000);   // 50000 + 20000 = 70000
    EXPECT_EQ(wide_sum[3], 70000);   // 60000 + 10000 = 70000

    // 加宽乘法 - uint16 → uint32
    vec8us c{200, 300, 400, 500, 200, 300, 400, 500};
    vec8us d{300, 400, 300, 200, 300, 400, 300, 200};

    auto wide_mul = mul_wide(c, d);
    EXPECT_EQ(wide_mul[0], 60000);   // 200 * 300 = 60000
    EXPECT_EQ(wide_mul[1], 120000);  // 300 * 400 = 120000
    EXPECT_EQ(wide_mul[2], 120000);  // 400 * 300 = 120000
    EXPECT_EQ(wide_mul[3], 100000);  // 500 * 200 = 100000

    std::cout << "uint16 widening results: " << wide_sum[1] << ", " << wide_mul[1] << std::endl;
}

TEST_F(OverflowTest, SignedIntegerOverflow) {
    // 有符号整数的溢出处理
    vec16b a{100, -100, 127, -128, 100, -100, 127, -128,
             100, -100, 127, -128, 100, -100, 127, -128};
    vec16b b{50, -50, 1, -1, 50, -50, 1, -1,
             50, -50, 1, -1, 50, -50, 1, -1};

    // 饱和加法 - 限制在 [-128, 127]
    vec16b sat_sum = add_sat(a, b);
    EXPECT_EQ(sat_sum[0], 127);   // min(100 + 50, 127) = 127 (饱和到上限)
    EXPECT_EQ(sat_sum[1], -128);  // max(-100 + (-50), -128) = -128 (饱和到下限)
    EXPECT_EQ(sat_sum[2], 127);   // min(127 + 1, 127) = 127 (已经在上限)
    EXPECT_EQ(sat_sum[3], -128);  // max(-128 + (-1), -128) = -128 (饱和到下限)

    // 饱和减法 - 限制在 [-128, 127]
    vec16b sat_sub = sub_sat(a, b);
    EXPECT_EQ(sat_sub[0], 50);    // 100 - 50 = 50
    EXPECT_EQ(sat_sub[1], -50);   // -100 - (-50) = -50
    EXPECT_EQ(sat_sub[2], 126);   // 127 - 1 = 126
    EXPECT_EQ(sat_sub[3], -127);  // -128 - (-1) = -127

    std::cout << "Signed saturation results: " << (int)sat_sum[0] << ", "
              << (int)sat_sum[1] << std::endl;
}

// 边界值测试
TEST_F(OverflowTest, BoundaryValues) {
    // 测试各种边界情况
    vec16ub max_vals{255, 255, 255, 255, 255, 255, 255, 255,
                     255, 255, 255, 255, 255, 255, 255, 255};
    vec16ub min_vals{0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0};
    vec16ub ones{1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1};

    // 最大值 + 1 (饱和)
    auto max_plus_one = add_sat(max_vals, ones);
    EXPECT_EQ(max_plus_one[0], 255);

    // 最小值 - 1 (饱和)
    auto min_minus_one = sub_sat(min_vals, ones);
    EXPECT_EQ(min_minus_one[0], 0);

    // 加宽运算应该正确处理边界值
    auto wide_max = add_wide(max_vals, ones);
    EXPECT_EQ(wide_max[0], 256);
}