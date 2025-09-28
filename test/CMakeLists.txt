# 添加 googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest)

# 包含 googletest 头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest/googletest/include)

add_subdirectory(unit_tests)
add_subdirectory(benchmarks)