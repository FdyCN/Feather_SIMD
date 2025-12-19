cmake_minimum_required(VERSION 3.12)
project(TinySimdEngine
    VERSION 1.0.0
    DESCRIPTION "High-performance cross-platform SIMD vector computation engine"
    LANGUAGES CXX
)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建类型配置
# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项 - 分平台设置
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang 编译选项
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # MSVC 编译选项
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Zi /DNDEBUG")
endif()

# 警告选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-variable
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(
        /W4
        /wd4100  # unreferenced formal parameter
        /wd4101  # unreferenced local variable
    )
    # 确保 MSVC 使用 C++11
    add_compile_options(/std:c++11)
endif()

# SIMD指令集检测和启用
include(CheckCXXCompilerFlag)

# 检测并启用SIMD指令集
function(enable_simd_instructions)
    # ARM NEON检测
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
        message(STATUS "Detected ARM processor, enabling NEON")
        add_compile_definitions(TINY_SIMD_ARM_NEON)

        # 对于某些编译器，可能需要显式启用NEON
        check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
        if(COMPILER_SUPPORTS_NEON)
            add_compile_options(-mfpu=neon)
        endif()

        # 检测 FP16 支持（Apple Silicon / ARMv8.2+）
        check_cxx_compiler_flag("-march=armv8.2-a+fp16" COMPILER_SUPPORTS_FP16)
        if(COMPILER_SUPPORTS_FP16)
            message(STATUS "ARM FP16 vector arithmetic support detected")
            add_compile_options(-march=armv8.2-a+fp16)
        else()
            # 尝试其他方式启用 FP16
            check_cxx_compiler_flag("-mfp16-format=ieee" COMPILER_SUPPORTS_FP16_IEEE)
            if(COMPILER_SUPPORTS_FP16_IEEE)
                message(STATUS "ARM FP16 IEEE format support detected")
                add_compile_options(-mfp16-format=ieee)
            endif()
        endif()
    endif()

    # x86 SIMD指令集检测
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
        message(STATUS "Detected x86 processor, checking SIMD support")

        # 根据编译器选择SIMD标志
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            # MSVC 使用 /arch: 标志
            # 注意：x64 默认支持 SSE2，所以只需要检测 AVX/AVX2
            check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                message(STATUS "AVX2 support detected")
                add_compile_options(/arch:AVX2)
                add_compile_definitions(TINY_SIMD_X86_AVX2)
            else()
                check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)
                if(COMPILER_SUPPORTS_AVX)
                    message(STATUS "AVX support detected")
                    add_compile_options(/arch:AVX)
                    add_compile_definitions(TINY_SIMD_X86_AVX)
                else()
                    # x64 MSVC 默认启用 SSE2
                    message(STATUS "Using default SSE2 support (x64 default)")
                    add_compile_definitions(TINY_SIMD_X86_SSE)
                endif()
            endif()
        else()
            # GCC/Clang 使用 -m 标志
            check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                message(STATUS "AVX2 support detected")
                add_compile_options(-mavx2)
                add_compile_definitions(TINY_SIMD_X86_AVX2)
            else()
                check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
                if(COMPILER_SUPPORTS_AVX)
                    message(STATUS "AVX support detected")
                    add_compile_options(-mavx)
                    add_compile_definitions(TINY_SIMD_X86_AVX)
                else()
                    check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
                    if(COMPILER_SUPPORTS_SSE2)
                        message(STATUS "SSE2 support detected")
                        add_compile_options(-msse2)
                        add_compile_definitions(TINY_SIMD_X86_SSE)
                    endif()
                endif()
            endif()

            # 检测SSE4.1 (用于某些优化) - 仅 GCC/Clang
            check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)
            if(COMPILER_SUPPORTS_SSE41)
                add_compile_options(-msse4.1)
            endif()
        endif()
    endif()
endfunction()

enable_simd_instructions()

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Header-only库目标
add_library(tiny_simd_core INTERFACE)
target_include_directories(tiny_simd_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# 添加数学库目标 (将来实现)
add_library(tiny_math INTERFACE)
target_include_directories(tiny_math INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tiny_math INTERFACE tiny_simd_core)

# 添加CV算子库目标 (将来实现)
add_library(tiny_cv INTERFACE)
target_include_directories(tiny_cv INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tiny_cv INTERFACE tiny_math)

# 选项控制
option(TINY_SIMD_BUILD_EXAMPLES "Build examples" ON)
option(TINY_SIMD_BUILD_TESTS "Build tests" ON)
option(TINY_SIMD_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TINY_SIMD_WITH_OPENCV "Build tests with OpenCV for accuracy and performance comparison" ON)

# OpenCV 集成 (可选)
if(TINY_SIMD_WITH_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core imgproc)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
        message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
        add_compile_definitions(TINY_SIMD_HAS_OPENCV)
    else()
        message(WARNING "OpenCV not found. Install it with: brew install opencv (macOS) or apt-get install libopencv-dev (Linux)")
        message(WARNING "Continuing without OpenCV support...")
        set(TINY_SIMD_WITH_OPENCV OFF)
    endif()
endif()

# 测试
if(TINY_SIMD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# 安装配置
include(GNUInstallDirs)

install(
    DIRECTORY core/ math/ operators/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tiny_simd
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS tiny_simd_core tiny_math tiny_cv
    EXPORT TinySimdEngineTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 导出配置
install(EXPORT TinySimdEngineTargets
    FILE TinySimdEngineTargets.cmake
    NAMESPACE TinySimdEngine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TinySimdEngine
)

# 包配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "TinySimdEngineConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TinySimdEngineConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TinySimdEngineConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TinySimdEngine
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TinySimdEngineConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TinySimdEngineConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TinySimdEngine
)

# 打印配置信息
message(STATUS "=== Tiny SIMD Engine Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build examples: ${TINY_SIMD_BUILD_EXAMPLES}")
message(STATUS "Build tests: ${TINY_SIMD_BUILD_TESTS}")
message(STATUS "Build benchmarks: ${TINY_SIMD_BUILD_BENCHMARKS}")
message(STATUS "OpenCV support: ${TINY_SIMD_WITH_OPENCV}")
message(STATUS "=======================================")