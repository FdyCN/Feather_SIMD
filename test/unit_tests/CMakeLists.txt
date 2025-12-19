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

# 如果启用了 OpenCV，添加 OpenCV 支持
if(TINY_SIMD_WITH_OPENCV AND OpenCV_FOUND)
    target_include_directories(tiny_simd_unit_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(tiny_simd_unit_tests PRIVATE ${OpenCV_LIBS})
    message(STATUS "OpenCV tests enabled for unit tests")
endif()

# 统一设置输出目录，方便直接运行
# MSVC 不支持 C++11，最低是 C++14；其他编译器使用 C++11
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set_target_properties(tiny_simd_unit_tests
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test
            CXX_STANDARD 14
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
    )
else()
    set_target_properties(tiny_simd_unit_tests
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test
            CXX_STANDARD 11
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
    )
endif()

# 注册到 CTest，默认运行全部测试用例
add_test(NAME tiny_simd_unit_tests COMMAND tiny_simd_unit_tests)
