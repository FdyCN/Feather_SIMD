#include "core/tiny_simd.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace tiny_simd;

inline bool float_equal(float a, float b, float tolerance = 1e-4f) {
    return std::abs(a - b) < tolerance;
}

class Float2NeonTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(Float2NeonTest, BasicConstruction) {
    // 测试各种构造方式
    vec2f default_vec;  // 默认构造
    EXPECT_EQ(default_vec.size(), 2);

    vec2f scalar_vec(5.0f);  // 标量构造
    EXPECT_FLOAT_EQ(scalar_vec[0], 5.0f);
    EXPECT_FLOAT_EQ(scalar_vec[1], 5.0f);

    vec2f init_vec{3.0f, 4.0f};  // 初始化列表构造
    EXPECT_FLOAT_EQ(init_vec[0], 3.0f);
    EXPECT_FLOAT_EQ(init_vec[1], 4.0f);

    float data[2] = {1.0f, 2.0f};
    vec2f ptr_vec(data);  // 指针构造
    EXPECT_FLOAT_EQ(ptr_vec[0], 1.0f);
    EXPECT_FLOAT_EQ(ptr_vec[1], 2.0f);
}

TEST_F(Float2NeonTest, ArithmeticOperations) {
    vec2f a{1.0f, 2.0f};
    vec2f b{3.0f, 4.0f};

    // 加法
    vec2f sum = a + b;
    EXPECT_FLOAT_EQ(sum[0], 4.0f);
    EXPECT_FLOAT_EQ(sum[1], 6.0f);

    // 减法
    vec2f diff = b - a;
    EXPECT_FLOAT_EQ(diff[0], 2.0f);
    EXPECT_FLOAT_EQ(diff[1], 2.0f);

    // 乘法
    vec2f mul = a * b;
    EXPECT_FLOAT_EQ(mul[0], 3.0f);
    EXPECT_FLOAT_EQ(mul[1], 8.0f);

    // 除法
    vec2f div = b / a;
    EXPECT_TRUE(float_equal(div[0], 3.0f));
    EXPECT_TRUE(float_equal(div[1], 2.0f));

    // 标量运算
    vec2f scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled[0], 2.0f);
    EXPECT_FLOAT_EQ(scaled[1], 4.0f);

    // 一元负号
    vec2f neg = -a;
    EXPECT_FLOAT_EQ(neg[0], -1.0f);
    EXPECT_FLOAT_EQ(neg[1], -2.0f);
}

TEST_F(Float2NeonTest, ComparisonOperations) {
    vec2f a{1.0f, 2.0f};
    vec2f b{1.0f, 2.0f};
    vec2f c{3.0f, 4.0f};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST_F(Float2NeonTest, VectorMathFunctions) {
    vec2f a{3.0f, 4.0f};
    vec2f b{1.0f, 2.0f};

    // 点积
    float dp = dot(a, b);
    EXPECT_FLOAT_EQ(dp, 11.0f);  // 3*1 + 4*2 = 11

    // 长度
    float len = length(a);
    EXPECT_FLOAT_EQ(len, 5.0f);  // sqrt(3^2 + 4^2) = 5

    // 长度平方
    float len_sq = length_squared(a);
    EXPECT_FLOAT_EQ(len_sq, 25.0f);  // 3^2 + 4^2 = 25

    // 归一化
    vec2f normalized = normalize(a);
    EXPECT_TRUE(float_equal(normalized[0], 0.6f));  // 3/5
    EXPECT_TRUE(float_equal(normalized[1], 0.8f));  // 4/5

    // 距离
    float dist = distance(a, b);
    EXPECT_FLOAT_EQ(dist, std::sqrt(8.0f));  // sqrt((3-1)^2 + (4-2)^2) = sqrt(8)
}

TEST_F(Float2NeonTest, MinMaxAbsFunctions) {
    vec2f a{-2.0f, 5.0f};
    vec2f b{3.0f, 1.0f};

    // 最小值
    vec2f min_result = min(a, b);
    EXPECT_FLOAT_EQ(min_result[0], -2.0f);
    EXPECT_FLOAT_EQ(min_result[1], 1.0f);

    // 最大值
    vec2f max_result = max(a, b);
    EXPECT_FLOAT_EQ(max_result[0], 3.0f);
    EXPECT_FLOAT_EQ(max_result[1], 5.0f);

    // 绝对值
    vec2f abs_result = abs(a);
    EXPECT_FLOAT_EQ(abs_result[0], 2.0f);
    EXPECT_FLOAT_EQ(abs_result[1], 5.0f);

    // clamp
    vec2f min_bound{-1.0f, -1.0f};
    vec2f max_bound{3.0f, 3.0f};
    vec2f clamped = clamp(a, min_bound, max_bound);
    EXPECT_FLOAT_EQ(clamped[0], -1.0f);  // -2 clamped to [-1, 3] = -1
    EXPECT_FLOAT_EQ(clamped[1], 3.0f);   // 5 clamped to [-1, 3] = 3
}

TEST_F(Float2NeonTest, MemoryOperations) {
    alignas(8) float aligned_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};

    // 对齐加载
    vec2f loaded = vec2f::load_aligned(aligned_data);
    EXPECT_FLOAT_EQ(loaded[0], 10.0f);
    EXPECT_FLOAT_EQ(loaded[1], 20.0f);

    // 存储
    vec2f to_store{100.0f, 200.0f};
    to_store.store_aligned(aligned_data);
    EXPECT_FLOAT_EQ(aligned_data[0], 100.0f);
    EXPECT_FLOAT_EQ(aligned_data[1], 200.0f);
}

TEST_F(Float2NeonTest, SIMDOptimizationCheck) {
    // 验证SIMD优化标志
    #ifdef TINY_SIMD_ARM_NEON
    EXPECT_TRUE(vec2f::is_simd_optimized);
    std::cout << "vec2f SIMD optimized: " << vec2f::is_simd_optimized << std::endl;
    #else
    GTEST_SKIP() << "NEON not available on this platform";
    #endif
}

TEST_F(Float2NeonTest, LinearInterpolation) {
    vec2f a{0.0f, 0.0f};
    vec2f b{10.0f, 20.0f};

    // 线性插值
    vec2f mid = lerp(a, b, 0.5f);
    EXPECT_FLOAT_EQ(mid[0], 5.0f);
    EXPECT_FLOAT_EQ(mid[1], 10.0f);

    vec2f quarter = lerp(a, b, 0.25f);
    EXPECT_FLOAT_EQ(quarter[0], 2.5f);
    EXPECT_FLOAT_EQ(quarter[1], 5.0f);
}

TEST_F(Float2NeonTest, VectorProjection) {
    vec2f a{4.0f, 3.0f};
    vec2f b{1.0f, 0.0f};  // 单位向量 x 轴

    // 向量投影
    vec2f proj = project(a, b);
    EXPECT_FLOAT_EQ(proj[0], 4.0f);  // a在b上的投影
    EXPECT_FLOAT_EQ(proj[1], 0.0f);
}
