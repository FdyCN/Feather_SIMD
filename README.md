# Tiny SIMD Engine

高性能跨平台SIMD向量计算库，提供统一的API接口封装ARM NEON和x86 SSE/AVX指令集。

## 项目结构

```
tiny-simd-engine/
├── core/
│   └── tiny_simd.hpp           # 核心SIMD底层库
├── math/
│   └── tiny_math.hpp           # 数学基础库(线性代数、数值计算等)
├── operators/
│   └── tiny_cv.hpp             # 计算机视觉算子库
├── test/
│   ├── unit_tests/             # 单元测试(googletest)
│   ├── benchmarks/             # 性能基准测试
│   └── opencv_compare/         # 与OpenCV对比测试
├── examples/
│   ├── basic_usage.cpp         # 基础使用示例
│   ├── cv_demo.cpp             # CV算子演示
│   └── performance_demo.cpp    # 性能对比演示
└── docs/
    └── design.md               # 设计文档
```

## 架构分层

```
CV算子层 (tiny_cv.hpp)      - 图像处理、滤波、特征检测
    ↓
数学库层 (tiny_math.hpp)    - 线性代数、数值计算、变换
    ↓
SIMD层 (tiny_simd.hpp)      - 跨平台SIMD指令封装
    ↓
硬件层                      - ARM NEON / x86 SSE/AVX
```

## 快速开始

```cpp
#include "operators/tiny_cv.hpp"

// 基础SIMD向量运算
auto v1 = tiny_simd::vec4f{1.0f, 2.0f, 3.0f, 4.0f};
auto v2 = tiny_simd::vec4f{2.0f, 3.0f, 4.0f, 5.0f};
auto result = v1 + v2;

// CV算子使用
float* blurred = tiny_cv::gaussian_blur(image_data, width, height, 2.0f);
```

## 编译要求

- C++11或更高版本
- CMake 3.12或更高版本
- 支持的平台：
  - ARM: ARMv7/ARMv8 with NEON
  - x86: SSE2/AVX/AVX2

## 编译和测试

### 使用CMake构建

```bash
# 1. 创建构建目录
mkdir build && cd build

# 2. 配置项目 (自动检测SIMD指令集)
cmake ..

# 3. 构建项目
make -j$(nproc)

# 或者使用cmake统一构建命令
# cmake --build . --parallel
```

### 运行测试

```bash
# 运行单元测试
./bin/test/simd_basic_test

# 运行基础使用示例
./bin/basic_usage

# 运行SIMD性能演示
./bin/simd_demo
```

### 构建选项

```bash
# Debug模式构建
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 禁用示例构建
cmake -DTINY_SIMD_BUILD_EXAMPLES=OFF ..

# 禁用测试构建
cmake -DTINY_SIMD_BUILD_TESTS=OFF ..

# 禁用基准测试
cmake -DTINY_SIMD_BUILD_BENCHMARKS=OFF ..
```

### SIMD指令集支持检测

CMake会自动检测并启用支持的SIMD指令集：
- **ARM平台**: 自动启用NEON (如果支持)
- **x86平台**: 按优先级检测AVX2 > AVX > SSE2

运行测试时会显示当前平台的SIMD支持情况：
```
SIMD Support:
  NEON: 1    # 1表示支持，0表示不支持
  SSE:  0
  AVX:  0
  AVX2: 0
```

## 安装和集成

### 作为Header-Only库使用

```cpp
// 直接包含需要的头文件
#include "core/tiny_simd.hpp"    // 基础SIMD向量库
#include "math/tiny_math.hpp"    // 数学运算库 (待实现)
#include "operators/tiny_cv.hpp" // CV算子库 (待实现)

// 使用示例
using namespace tiny_simd;
vec4f a{1.0f, 2.0f, 3.0f, 4.0f};
vec4f b{2.0f, 3.0f, 4.0f, 5.0f};
vec4f result = a + b;  // 自动使用SIMD优化
```

### 使用CMake集成到项目

```cmake
# 在你的CMakeLists.txt中
add_subdirectory(path/to/tiny-simd-engine)

# 链接到你的目标
target_link_libraries(your_target
    TinySimdEngine::tiny_simd_core
    TinySimdEngine::tiny_math
    TinySimdEngine::tiny_cv
)
```

### 验证SIMD优化效果

运行示例程序查看SIMD优化状态：
```bash
# 查看当前平台SIMD支持
./bin/basic_usage

# 输出示例:
# Platform SIMD Support:
#   NEON: Yes          # ARM平台
#   SSE:  No
#   vec4f optimized: 1 # 1表示使用SIMD优化
```

## 性能目标

通过SIMD优化，预期相比标量实现：
- 浮点运算性能提升2-4倍
- 整数运算性能提升4-8倍
- 内存带宽利用率提升2-4倍

## 故障排除

### 常见编译问题

1. **CMake版本过低**
```bash
# 确保CMake版本 >= 3.12
cmake --version

# macOS升级CMake
brew upgrade cmake

# Ubuntu升级CMake
sudo apt update && sudo apt install cmake
```

2. **SIMD指令集未启用**
```bash
# 检查编译器支持
./bin/basic_usage | grep "SIMD Support"

# 如果显示全为"No"，可能需要：
# - 确认处理器支持SIMD指令集
# - 使用支持的编译器版本
# - 手动指定编译选项：cmake -DCMAKE_CXX_FLAGS="-march=native" ..
```

3. **在Docker或虚拟机中SIMD检测失败**
```bash
# 虚拟机可能不支持SIMD，会自动回退到标量实现
# 这是正常行为，功能仍然可用，只是性能不会有SIMD优化
```

### 性能验证

```bash
# 运行性能测试
./bin/simd_demo

# 如果SIMD优化正常，应该看到：
# - "SIMD-optimized implementation"
# - vec4f/vec4i显示"optimized: 1"
# - 部分测试显示性能提升
```