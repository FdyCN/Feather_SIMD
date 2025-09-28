#include "core/tiny_simd.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tiny_simd;

bool float_equal(float a, float b, float tolerance = 1e-6f) {
    return std::abs(a - b) < tolerance;
}

void test_platform_detection() {
    std::cout << "=== Platform Detection Test ===" << std::endl;

    std::cout << "SIMD Support:" << std::endl;
    std::cout << "  NEON: " << config::has_neon << std::endl;
    std::cout << "  SSE:  " << config::has_sse << std::endl;
    std::cout << "  AVX:  " << config::has_avx << std::endl;
    std::cout << "  AVX2: " << config::has_avx2 << std::endl;

    std::cout << "Vector Limits:" << std::endl;
    std::cout << "  float:  " << config::max_vector_size_float << std::endl;
    std::cout << "  double: " << config::max_vector_size_double << std::endl;
    std::cout << "  int32:  " << config::max_vector_size_int32 << std::endl;

    std::cout << "Test PASSED" << std::endl << std::endl;
}

void test_vector_construction() {
    std::cout << "=== Vector Construction Test ===" << std::endl;

    // 默认构造
    vec4f v1;

    // 标量构造
    vec4f v2(5.0f);
    for (int i = 0; i < 4; ++i) {
        assert(v2[i] == 5.0f);
    }

    // 参数包构造
    vec4f v3{1.0f, 2.0f, 3.0f, 4.0f};
    assert(v3[0] == 1.0f);
    assert(v3[1] == 2.0f);
    assert(v3[2] == 3.0f);
    assert(v3[3] == 4.0f);

    // 从指针构造
    float data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    vec4f v4(data);
    assert(v4[0] == 10.0f);
    assert(v4[1] == 20.0f);
    assert(v4[2] == 30.0f);
    assert(v4[3] == 40.0f);

    std::cout << "Construction optimized: " << vec4f::is_simd_optimized << std::endl;
    std::cout << "Test PASSED" << std::endl << std::endl;
}

void test_arithmetic_operations() {
    std::cout << "=== Arithmetic Operations Test ===" << std::endl;

    vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
    vec4f b{5.0f, 6.0f, 7.0f, 8.0f};

    // 加法
    vec4f sum = a + b;
    assert(sum[0] == 6.0f);
    assert(sum[1] == 8.0f);
    assert(sum[2] == 10.0f);
    assert(sum[3] == 12.0f);

    // 减法
    vec4f diff = b - a;
    assert(diff[0] == 4.0f);
    assert(diff[1] == 4.0f);
    assert(diff[2] == 4.0f);
    assert(diff[3] == 4.0f);

    // 乘法
    vec4f product = a * b;
    assert(product[0] == 5.0f);
    assert(product[1] == 12.0f);
    assert(product[2] == 21.0f);
    assert(product[3] == 32.0f);

    // 除法
    vec4f quotient = b / a;
    assert(float_equal(quotient[0], 5.0f));
    assert(float_equal(quotient[1], 3.0f));
    assert(float_equal(quotient[2], 7.0f/3.0f));
    assert(float_equal(quotient[3], 2.0f));

    // 标量运算
    vec4f scaled = a * 2.0f;
    assert(scaled[0] == 2.0f);
    assert(scaled[1] == 4.0f);
    assert(scaled[2] == 6.0f);
    assert(scaled[3] == 8.0f);

    std::cout << "Arithmetic optimized: " << vec4f::is_simd_optimized << std::endl;
    std::cout << "Test PASSED" << std::endl << std::endl;
}

void test_vector_functions() {
    std::cout << "=== Vector Functions Test ===" << std::endl;

    vec3f v{3.0f, 4.0f, 0.0f};

    // 长度测试
    float len_sq = length_squared(v);
    assert(float_equal(len_sq, 25.0f));

    float len = length(v);
    assert(float_equal(len, 5.0f));

    // 归一化测试
    vec3f normalized = normalize(v);
    assert(float_equal(normalized[0], 0.6f));
    assert(float_equal(normalized[1], 0.8f));
    assert(float_equal(normalized[2], 0.0f));

    // 点积测试
    vec3f v1{1.0f, 2.0f, 3.0f};
    vec3f v2{4.0f, 5.0f, 6.0f};
    float dot_result = dot(v1, v2);
    assert(float_equal(dot_result, 32.0f)); // 1*4 + 2*5 + 3*6 = 32

    // 叉积测试
    vec3f cross_result = cross(v1, v2);
    assert(float_equal(cross_result[0], -3.0f));  // 2*6 - 3*5 = -3
    assert(float_equal(cross_result[1], 6.0f));   // 3*4 - 1*6 = 6
    assert(float_equal(cross_result[2], -3.0f));  // 1*5 - 2*4 = -3

    // 距离测试
    float dist = distance(v1, v2);
    assert(float_equal(dist, std::sqrt(27.0f))); // sqrt((4-1)^2 + (5-2)^2 + (6-3)^2)

    std::cout << "vec3f optimized: " << vec3f::is_simd_optimized << std::endl;
    std::cout << "Test PASSED" << std::endl << std::endl;
}

void test_memory_operations() {
    std::cout << "=== Memory Operations Test ===" << std::endl;

    // 对齐内存测试
    alignas(config::simd_alignment) float aligned_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    vec4f v1 = vec4f::load_aligned(aligned_data);
    assert(v1[0] == 1.0f);
    assert(v1[1] == 2.0f);
    assert(v1[2] == 3.0f);
    assert(v1[3] == 4.0f);

    // 存储测试
    vec4f v2{10.0f, 20.0f, 30.0f, 40.0f};
    v2.store_aligned(aligned_data);
    assert(aligned_data[0] == 10.0f);
    assert(aligned_data[1] == 20.0f);
    assert(aligned_data[2] == 30.0f);
    assert(aligned_data[3] == 40.0f);

    std::cout << "Memory alignment: " << config::simd_alignment << " bytes" << std::endl;
    std::cout << "Test PASSED" << std::endl << std::endl;
}

void test_different_types() {
    std::cout << "=== Different Types Test ===" << std::endl;

    // 整数向量测试
    vec4i int_vec{1, 2, 3, 4};
    vec4i int_sum = int_vec + int_vec;
    assert(int_sum[0] == 2);
    assert(int_sum[1] == 4);
    assert(int_sum[2] == 6);
    assert(int_sum[3] == 8);

    // 双精度向量测试 (如果支持)
    vec2d double_vec{1.0, 2.0};
    vec2d double_sum = double_vec + double_vec;
    assert(double_sum[0] == 2.0);
    assert(double_sum[1] == 4.0);

    std::cout << "vec4i optimized: " << vec4i::is_simd_optimized << std::endl;
    std::cout << "vec2d optimized: " << vec2d::is_simd_optimized << std::endl;
    std::cout << "Test PASSED" << std::endl << std::endl;
}

void test_edge_cases() {
    std::cout << "=== Edge Cases Test ===" << std::endl;

    // 零向量测试
    vec3f zero{0.0f, 0.0f, 0.0f};
    vec3f normalized_zero = normalize(zero);
    assert(normalized_zero[0] == 0.0f);
    assert(normalized_zero[1] == 0.0f);
    assert(normalized_zero[2] == 0.0f);

    // 负数测试
    vec4f negative{-1.0f, -2.0f, -3.0f, -4.0f};
    vec4f abs_negative = abs(negative);
    assert(abs_negative[0] == 1.0f);
    assert(abs_negative[1] == 2.0f);
    assert(abs_negative[2] == 3.0f);
    assert(abs_negative[3] == 4.0f);

    // 边界值测试
    vec4f small{1e-5f, 1e-5f, 1e-5f, 1e-5f};
    vec4f large{1e5f, 1e5f, 1e5f, 1e5f};
    vec4f mixed = small + large;
    std::cout << "small[0]: " << small[0] << ", large[0]: " << large[0] << ", mixed[0]: " << mixed[0] << std::endl;
    assert(mixed[0] >= large[0]);  // 应该是大于等于large的值

    std::cout << "Test PASSED" << std::endl << std::endl;
}

int main() {
    std::cout << "=== Tiny SIMD Engine Basic Tests ===" << std::endl;
    std::cout << "Running on " << (vec4f::is_simd_optimized ? "SIMD-optimized" : "scalar fallback") << " implementation" << std::endl;
    std::cout << std::endl;

    try {
        test_platform_detection();
        test_vector_construction();
        test_arithmetic_operations();
        test_vector_functions();
        test_memory_operations();
        test_different_types();
        test_edge_cases();

        std::cout << "=== ALL TESTS PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}