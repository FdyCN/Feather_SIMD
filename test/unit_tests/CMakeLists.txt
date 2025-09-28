# 创建测试可执行文件
add_executable(simd_basic_test simd_basic_test.cpp)

# 链接库
target_link_libraries(simd_basic_test
    tiny_simd_core
    gtest
    gtest_main
)

# 设置输出目录
set_target_properties(simd_basic_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test
)

# 添加测试到 CTest
add_test(NAME simd_basic_test COMMAND simd_basic_test)