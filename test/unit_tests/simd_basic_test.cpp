#include "core/tiny_simd.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace tiny_simd;

inline bool float_equal(float a, float b, float tolerance = 1e-6f) {
    return std::abs(a - b) < tolerance;
}

TEST(PlatformDetectionTest, SimdSupport) {
    // 这些测试只是确保配置值是布尔值且可访问
    EXPECT_TRUE(config::has_neon == true || config::has_neon == false);
    EXPECT_TRUE(config::has_sse == true || config::has_sse == false);
    EXPECT_TRUE(config::has_avx == true || config::has_avx == false);
    EXPECT_TRUE(config::has_avx2 == true || config::has_avx2 == false);

    // 验证向量长度限制为正数
    EXPECT_GT(config::max_vector_size_float, 0u);
    EXPECT_GT(config::max_vector_size_double, 0u);
    EXPECT_GT(config::max_vector_size_int32, 0u);

    // 输出平台信息以便调试
    std::cout << "SIMD Support - NEON: " << config::has_neon
              << ", SSE: " << config::has_sse
              << ", AVX: " << config::has_avx
              << ", AVX2: " << config::has_avx2 << std::endl;
}

TEST(VectorConstructionTest, DefaultConstruction) {
    vec4f v1;
    // 默认构造函数应该工作（不测试具体值，因为可能未初始化）
    EXPECT_EQ(v1.size(), 4u);
}

TEST(VectorConstructionTest, ScalarConstruction) {
    vec4f v2(5.0f);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(v2[i], 5.0f);
    }
}

TEST(VectorConstructionTest, InitializerListConstruction) {
    vec4f v3{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_FLOAT_EQ(v3[0], 1.0f);
    EXPECT_FLOAT_EQ(v3[1], 2.0f);
    EXPECT_FLOAT_EQ(v3[2], 3.0f);
    EXPECT_FLOAT_EQ(v3[3], 4.0f);
}

TEST(VectorConstructionTest, PointerConstruction) {
    float data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    vec4f v4(data);
    EXPECT_FLOAT_EQ(v4[0], 10.0f);
    EXPECT_FLOAT_EQ(v4[1], 20.0f);
    EXPECT_FLOAT_EQ(v4[2], 30.0f);
    EXPECT_FLOAT_EQ(v4[3], 40.0f);
}

TEST(ArithmeticOperationsTest, Addition) {
    vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
    vec4f b{5.0f, 6.0f, 7.0f, 8.0f};

    vec4f sum = a + b;
    EXPECT_FLOAT_EQ(sum[0], 6.0f);
    EXPECT_FLOAT_EQ(sum[1], 8.0f);
    EXPECT_FLOAT_EQ(sum[2], 10.0f);
    EXPECT_FLOAT_EQ(sum[3], 12.0f);
}

TEST(ArithmeticOperationsTest, Subtraction) {
    vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
    vec4f b{5.0f, 6.0f, 7.0f, 8.0f};

    vec4f diff = b - a;
    EXPECT_FLOAT_EQ(diff[0], 4.0f);
    EXPECT_FLOAT_EQ(diff[1], 4.0f);
    EXPECT_FLOAT_EQ(diff[2], 4.0f);
    EXPECT_FLOAT_EQ(diff[3], 4.0f);
}

TEST(ArithmeticOperationsTest, Multiplication) {
    vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
    vec4f b{5.0f, 6.0f, 7.0f, 8.0f};

    vec4f product = a * b;
    EXPECT_FLOAT_EQ(product[0], 5.0f);
    EXPECT_FLOAT_EQ(product[1], 12.0f);
    EXPECT_FLOAT_EQ(product[2], 21.0f);
    EXPECT_FLOAT_EQ(product[3], 32.0f);
}

TEST(ArithmeticOperationsTest, Division) {
    vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
    vec4f b{5.0f, 6.0f, 7.0f, 8.0f};

    vec4f quotient = b / a;
    // 使用更宽松的tolerance，因为NEON除法使用近似算法
    EXPECT_TRUE(float_equal(quotient[0], 5.0f, 1e-3f));
    EXPECT_TRUE(float_equal(quotient[1], 3.0f, 1e-3f));
    EXPECT_TRUE(float_equal(quotient[2], 7.0f/3.0f, 1e-3f));
    EXPECT_TRUE(float_equal(quotient[3], 2.0f, 1e-3f));
}

TEST(ArithmeticOperationsTest, ScalarOperations) {
    vec4f a{1.0f, 2.0f, 3.0f, 4.0f};

    vec4f scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled[0], 2.0f);
    EXPECT_FLOAT_EQ(scaled[1], 4.0f);
    EXPECT_FLOAT_EQ(scaled[2], 6.0f);
    EXPECT_FLOAT_EQ(scaled[3], 8.0f);
}

TEST(VectorFunctionsTest, LengthOperations) {
    vec3f v{3.0f, 4.0f, 0.0f};

    float len_sq = length_squared(v);
    EXPECT_TRUE(float_equal(len_sq, 25.0f));

    float len = length(v);
    EXPECT_TRUE(float_equal(len, 5.0f));
}

TEST(VectorFunctionsTest, Normalization) {
    vec3f v{3.0f, 4.0f, 0.0f};
    vec3f normalized = normalize(v);
    EXPECT_TRUE(float_equal(normalized[0], 0.6f));
    EXPECT_TRUE(float_equal(normalized[1], 0.8f));
    EXPECT_TRUE(float_equal(normalized[2], 0.0f));
}

TEST(VectorFunctionsTest, DotProduct) {
    vec3f v1{1.0f, 2.0f, 3.0f};
    vec3f v2{4.0f, 5.0f, 6.0f};
    float dot_result = dot(v1, v2);
    EXPECT_TRUE(float_equal(dot_result, 32.0f)); // 1*4 + 2*5 + 3*6 = 32
}

TEST(VectorFunctionsTest, CrossProduct) {
    vec3f v1{1.0f, 2.0f, 3.0f};
    vec3f v2{4.0f, 5.0f, 6.0f};
    vec3f cross_result = cross(v1, v2);
    EXPECT_TRUE(float_equal(cross_result[0], -3.0f));  // 2*6 - 3*5 = -3
    EXPECT_TRUE(float_equal(cross_result[1], 6.0f));   // 3*4 - 1*6 = 6
    EXPECT_TRUE(float_equal(cross_result[2], -3.0f));  // 1*5 - 2*4 = -3
}

TEST(VectorFunctionsTest, Distance) {
    vec3f v1{1.0f, 2.0f, 3.0f};
    vec3f v2{4.0f, 5.0f, 6.0f};
    float dist = distance(v1, v2);
    EXPECT_TRUE(float_equal(dist, std::sqrt(27.0f))); // sqrt((4-1)^2 + (5-2)^2 + (6-3)^2)
}

TEST(MemoryOperationsTest, AlignedLoadStore) {
    alignas(config::simd_alignment) float aligned_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    vec4f v1 = vec4f::load_aligned(aligned_data);
    EXPECT_FLOAT_EQ(v1[0], 1.0f);
    EXPECT_FLOAT_EQ(v1[1], 2.0f);
    EXPECT_FLOAT_EQ(v1[2], 3.0f);
    EXPECT_FLOAT_EQ(v1[3], 4.0f);

    vec4f v2{10.0f, 20.0f, 30.0f, 40.0f};
    v2.store_aligned(aligned_data);
    EXPECT_FLOAT_EQ(aligned_data[0], 10.0f);
    EXPECT_FLOAT_EQ(aligned_data[1], 20.0f);
    EXPECT_FLOAT_EQ(aligned_data[2], 30.0f);
    EXPECT_FLOAT_EQ(aligned_data[3], 40.0f);
}

TEST(DifferentTypesTest, IntegerVectors) {
    vec4i int_vec{1, 2, 3, 4};
    vec4i int_sum = int_vec + int_vec;
    EXPECT_EQ(int_sum[0], 2);
    EXPECT_EQ(int_sum[1], 4);
    EXPECT_EQ(int_sum[2], 6);
    EXPECT_EQ(int_sum[3], 8);
}

TEST(DifferentTypesTest, DoubleVectors) {
    vec2d double_vec{1.0, 2.0};
    vec2d double_sum = double_vec + double_vec;
    EXPECT_DOUBLE_EQ(double_sum[0], 2.0);
    EXPECT_DOUBLE_EQ(double_sum[1], 4.0);
}

TEST(EdgeCasesTest, ZeroVectorNormalization) {
    vec3f zero{0.0f, 0.0f, 0.0f};
    vec3f normalized_zero = normalize(zero);
    EXPECT_FLOAT_EQ(normalized_zero[0], 0.0f);
    EXPECT_FLOAT_EQ(normalized_zero[1], 0.0f);
    EXPECT_FLOAT_EQ(normalized_zero[2], 0.0f);
}

TEST(EdgeCasesTest, AbsoluteValue) {
    vec4f negative{-1.0f, -2.0f, -3.0f, -4.0f};
    vec4f abs_negative = abs(negative);
    EXPECT_FLOAT_EQ(abs_negative[0], 1.0f);
    EXPECT_FLOAT_EQ(abs_negative[1], 2.0f);
    EXPECT_FLOAT_EQ(abs_negative[2], 3.0f);
    EXPECT_FLOAT_EQ(abs_negative[3], 4.0f);
}

TEST(EdgeCasesTest, BoundaryValues) {
    vec4f small{1e-5f, 1e-5f, 1e-5f, 1e-5f};
    vec4f large{1e5f, 1e5f, 1e5f, 1e5f};
    vec4f mixed = small + large;
    EXPECT_GE(mixed[0], large[0]);
}

// Google Test 会自动提供 main 函数
