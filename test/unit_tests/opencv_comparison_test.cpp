/**
 * @file opencv_comparison_test.cpp
 * @brief OpenCV comparison tests for accuracy and performance validation
 *
 * This test file demonstrates how to use OpenCV as a reference implementation
 * to validate the accuracy and performance of SIMD operations.
 *
 * Build with: cmake -DTINY_SIMD_WITH_OPENCV=ON ..
 */

#include <gtest/gtest.h>
#include "core/tiny_simd.hpp"
#include <vector>
#include <cmath>
#include <chrono>

#ifdef TINY_SIMD_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace tiny_simd;

/**
 * @brief Helper class for accuracy and performance comparison
 */
class OpenCVComparison : public ::testing::Test {
protected:
    static const int WARMUP_ITERATIONS = 100;
    static const int BENCHMARK_ITERATIONS = 10000;

    // Use inline constant instead of static constexpr for floating point
    float epsilon() const { return 1e-5f; }

    void SetUp() override {
        // Initialize test data
        test_size_ = 1024;
        a_data_.resize(test_size_);
        b_data_.resize(test_size_);
        result_simd_.resize(test_size_);
        result_opencv_.resize(test_size_);

        // Fill with random data
        for (int i = 0; i < test_size_; ++i) {
            a_data_[i] = static_cast<float>(i) * 0.1f;
            b_data_[i] = static_cast<float>(test_size_ - i) * 0.1f;
        }
    }

    template<typename Func>
    double measureTime(Func&& func, int iterations = BENCHMARK_ITERATIONS) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            func();
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    }

    void compareResults(const std::vector<float>& simd_result,
                       const std::vector<float>& opencv_result,
                       const std::string& operation_name) {
        ASSERT_EQ(simd_result.size(), opencv_result.size());

        const float EPSILON = epsilon();
        float max_error = 0.0f;
        int error_count = 0;

        for (size_t i = 0; i < simd_result.size(); ++i) {
            float error = std::abs(simd_result[i] - opencv_result[i]);
            max_error = std::max(max_error, error);

            if (error > EPSILON) {
                error_count++;
                if (error_count <= 5) {  // Print first 5 errors
                    std::cout << operation_name << " Error at [" << i << "]: "
                             << "SIMD=" << simd_result[i]
                             << ", OpenCV=" << opencv_result[i]
                             << ", diff=" << error << std::endl;
                }
            }
        }

        std::cout << operation_name << " - Max error: " << max_error
                  << ", Error count: " << error_count << std::endl;

        EXPECT_LT(max_error, EPSILON) << "Maximum error exceeds threshold";
        EXPECT_EQ(error_count, 0) << "Found " << error_count << " elements with errors";
    }

    int test_size_;
    std::vector<float> a_data_;
    std::vector<float> b_data_;
    std::vector<float> result_simd_;
    std::vector<float> result_opencv_;
};

/**
 * @brief Test vector addition accuracy
 */
TEST_F(OpenCVComparison, VectorAdditionAccuracy) {
    // SIMD implementation
    for (int i = 0; i < test_size_; i += 4) {
        vec4f a(&a_data_[i]);
        vec4f b(&b_data_[i]);
        vec4f result = a + b;
        result.store(&result_simd_[i]);
    }

    // OpenCV implementation
    cv::Mat mat_a(1, test_size_, CV_32F, a_data_.data());
    cv::Mat mat_b(1, test_size_, CV_32F, b_data_.data());
    cv::Mat mat_result;
    cv::add(mat_a, mat_b, mat_result);
    std::memcpy(result_opencv_.data(), mat_result.data, test_size_ * sizeof(float));

    compareResults(result_simd_, result_opencv_, "Vector Addition");
}

/**
 * @brief Test vector subtraction accuracy
 */
TEST_F(OpenCVComparison, VectorSubtractionAccuracy) {
    // SIMD implementation
    for (int i = 0; i < test_size_; i += 4) {
        vec4f a(&a_data_[i]);
        vec4f b(&b_data_[i]);
        vec4f result = a - b;
        result.store(&result_simd_[i]);
    }

    // OpenCV implementation
    cv::Mat mat_a(1, test_size_, CV_32F, a_data_.data());
    cv::Mat mat_b(1, test_size_, CV_32F, b_data_.data());
    cv::Mat mat_result;
    cv::subtract(mat_a, mat_b, mat_result);
    std::memcpy(result_opencv_.data(), mat_result.data, test_size_ * sizeof(float));

    compareResults(result_simd_, result_opencv_, "Vector Subtraction");
}

/**
 * @brief Test vector multiplication accuracy
 */
TEST_F(OpenCVComparison, VectorMultiplicationAccuracy) {
    // SIMD implementation
    for (int i = 0; i < test_size_; i += 4) {
        vec4f a(&a_data_[i]);
        vec4f b(&b_data_[i]);
        vec4f result = a * b;
        result.store(&result_simd_[i]);
    }

    // OpenCV implementation
    cv::Mat mat_a(1, test_size_, CV_32F, a_data_.data());
    cv::Mat mat_b(1, test_size_, CV_32F, b_data_.data());
    cv::Mat mat_result;
    cv::multiply(mat_a, mat_b, mat_result);
    std::memcpy(result_opencv_.data(), mat_result.data, test_size_ * sizeof(float));

    compareResults(result_simd_, result_opencv_, "Vector Multiplication");
}

/**
 * @brief Benchmark vector addition performance
 */
TEST_F(OpenCVComparison, VectorAdditionPerformance) {
    // SIMD benchmark
    double simd_time = measureTime([&]() {
        for (int i = 0; i < test_size_; i += 4) {
            vec4f a(&a_data_[i]);
            vec4f b(&b_data_[i]);
            vec4f result = a + b;
            result.store(&result_simd_[i]);
        }
    });

    // OpenCV benchmark
    cv::Mat mat_a(1, test_size_, CV_32F, a_data_.data());
    cv::Mat mat_b(1, test_size_, CV_32F, b_data_.data());
    cv::Mat mat_result;

    double opencv_time = measureTime([&]() {
        cv::add(mat_a, mat_b, mat_result);
    });

    std::cout << "\n=== Vector Addition Performance ===" << std::endl;
    std::cout << "SIMD:   " << simd_time << " ms/iter" << std::endl;
    std::cout << "OpenCV: " << opencv_time << " ms/iter" << std::endl;
    std::cout << "Speedup: " << (opencv_time / simd_time) << "x" << std::endl;
}

/**
 * @brief Benchmark vector multiplication performance
 */
TEST_F(OpenCVComparison, VectorMultiplicationPerformance) {
    // SIMD benchmark
    double simd_time = measureTime([&]() {
        for (int i = 0; i < test_size_; i += 4) {
            vec4f a(&a_data_[i]);
            vec4f b(&b_data_[i]);
            vec4f result = a * b;
            result.store(&result_simd_[i]);
        }
    });

    // OpenCV benchmark
    cv::Mat mat_a(1, test_size_, CV_32F, a_data_.data());
    cv::Mat mat_b(1, test_size_, CV_32F, b_data_.data());
    cv::Mat mat_result;

    double opencv_time = measureTime([&]() {
        cv::multiply(mat_a, mat_b, mat_result);
    });

    std::cout << "\n=== Vector Multiplication Performance ===" << std::endl;
    std::cout << "SIMD:   " << simd_time << " ms/iter" << std::endl;
    std::cout << "OpenCV: " << opencv_time << " ms/iter" << std::endl;
    std::cout << "Speedup: " << (opencv_time / simd_time) << "x" << std::endl;
}

#else

// Placeholder test when OpenCV is not available
TEST(OpenCVComparison, NotAvailable) {
    GTEST_SKIP() << "OpenCV support not enabled. Build with -DTINY_SIMD_WITH_OPENCV=ON";
}

#endif  // TINY_SIMD_HAS_OPENCV
