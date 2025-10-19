# 聚合所有单元测试源文件，自动发现目录下的所有 .cpp
file(GLOB TINY_SIMD_UNIT_TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# 创建单一的测试可执行文件，使用同一个 GoogleTest main
add_executable(tiny_simd_unit_tests
    ${TINY_SIMD_UNIT_TEST_SOURCES}
)

# 链接核心库和 GoogleTest
target_link_libraries(tiny_simd_unit_tests
    PRIVATE
        tiny_simd_core
        gtest
        gtest_main
)

# 统一设置输出目录，方便直接运行
set_target_properties(tiny_simd_unit_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test
)

# 注册到 CTest，默认运行全部测试用例
add_test(NAME tiny_simd_unit_tests COMMAND tiny_simd_unit_tests)
