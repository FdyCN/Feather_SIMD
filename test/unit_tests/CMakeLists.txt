add_executable(simd_basic_test simd_basic_test.cpp)
target_link_libraries(simd_basic_test tiny_simd_core)

# 设置输出目录
set_target_properties(simd_basic_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test
)