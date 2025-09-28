# 测试框架说明

本项目使用 Google Test 作为单元测试框架，通过 git submodule 集成，无需系统级安装。

## 重要说明

⚠️ **C++11 兼容性**: 本项目使用 C++11 标准，因此集成了 **GoogleTest v1.12.0**，这是最后一个完全支持 C++11 的版本。

- ✅ **GoogleTest v1.12.0**: 支持 C++11（当前使用）
- ❌ **GoogleTest v1.13.0+**: 要求 C++14 或更高版本

## 项目结构

```
test/
├── README.md                     # 本文件
├── CMakeLists.txt               # 测试构建配置
├── unit_tests/                  # 单元测试
│   ├── CMakeLists.txt
│   └── simd_basic_test.cpp     # SIMD核心功能测试
├── benchmarks/                  # 性能基准测试
└── opencv_compare/             # OpenCV对比测试
```

## GoogleTest 集成

### 自动集成 (推荐)

项目已经通过 git submodule 集成了 GoogleTest，无需手动安装：

```bash
# 克隆项目时包含子模块
git clone --recursive https://github.com/your-repo/claude-tiny-engine.git

# 如果已经克隆，初始化子模块
git submodule update --init --recursive
```

### 构建和运行测试

```bash
# 配置和构建
mkdir -p build
cd build
cmake ..
make

# 运行所有测试 (三种方式)
./bin/test/simd_basic_test           # 直接运行可执行文件
ctest                                # 使用 CTest
cmake --build . --target test       # 使用 CMake 测试目标

# 从项目根目录运行测试
cd ..
ctest --test-dir build              # 指定构建目录
cmake --build build --target test   # 使用 CMake
```

### 详细测试输出

```bash
# 详细的 GoogleTest 输出
./bin/test/simd_basic_test --gtest_verbose

# 详细的 CTest 输出
ctest --verbose

# 只运行特定测试
./bin/test/simd_basic_test --gtest_filter="VectorConstructionTest*"
```

## 当前测试覆盖

### SIMD 基础测试 (`simd_basic_test.cpp`)

- **平台检测测试** (`PlatformDetectionTest`)
  - SIMD 指令集支持检测 (NEON, SSE, AVX, AVX2)
  - 向量大小限制验证

- **向量构造测试** (`VectorConstructionTest`)
  - 默认构造函数
  - 标量构造函数
  - 初始化列表构造
  - 指针构造

- **算术运算测试** (`ArithmeticOperationsTest`)
  - 向量加法、减法、乘法、除法
  - 标量运算

- **向量函数测试** (`VectorFunctionsTest`)
  - 长度计算 (`length`, `length_squared`)
  - 向量归一化 (`normalize`)
  - 点积 (`dot`)
  - 叉积 (`cross`)
  - 距离计算 (`distance`)

- **内存操作测试** (`MemoryOperationsTest`)
  - 对齐内存加载和存储
  - SIMD 对齐要求验证

- **不同类型测试** (`DifferentTypesTest`)
  - 整数向量 (`vec4i`)
  - 双精度向量 (`vec2d`)

- **边界情况测试** (`EdgeCasesTest`)
  - 零向量处理
  - 负数绝对值
  - 数值边界值

## SIMD 优化验证

测试会自动检测和验证 SIMD 优化：

- **ARM NEON**: `vec4f` 和 `vec4i` 完全优化
- **x86 SSE/AVX**: 根据编译时标志启用
- **标量回退**: 当 SIMD 不可用时自动使用

## 添加新测试

### 创建新的测试文件

1. 在 `test/unit_tests/` 目录下创建新的 `.cpp` 文件
2. 包含必要的头文件：
   ```cpp
   #include "core/tiny_simd.hpp"
   #include <gtest/gtest.h>
   ```

3. 编写测试用例：
   ```cpp
   TEST(TestSuiteName, TestCaseName) {
       // 测试代码
       EXPECT_EQ(expected, actual);
       EXPECT_FLOAT_EQ(expected_float, actual_float);
       EXPECT_TRUE(condition);
   }
   ```

4. 更新 `test/unit_tests/CMakeLists.txt`：
   ```cmake
   add_executable(new_test new_test.cpp)
   target_link_libraries(new_test tiny_simd_core gtest gtest_main)
   add_test(NAME new_test COMMAND new_test)
   ```

### GoogleTest 断言

- `EXPECT_EQ(a, b)` - 相等
- `EXPECT_FLOAT_EQ(a, b)` - 浮点数相等
- `EXPECT_DOUBLE_EQ(a, b)` - 双精度相等
- `EXPECT_TRUE(condition)` - 条件为真
- `EXPECT_GT(a, b)` - 大于
- `EXPECT_GE(a, b)` - 大于等于

## 性能测试

基准测试位于 `benchmarks/` 目录，可以用于：

- SIMD vs 标量性能对比
- 不同向量大小的性能测试
- 内存对齐的性能影响

## CI/CD 集成

项目测试框架支持持续集成：

```yaml
# GitHub Actions 示例
- name: Build and Test
  run: |
    cmake -B build
    cmake --build build
    ctest --test-dir build --output-on-failure
```

## 故障排除

### 常见问题

1. **找不到测试配置文件**
   ```bash
   # 错误: 在项目根目录运行 ctest
   # 正确: 指定构建目录
   ctest --test-dir build
   ```

2. **SIMD 测试失败**
   - 检查目标平台的 SIMD 支持
   - 验证编译器标志 (`-mavx2`, `-mfpu=neon`)

3. **精度问题**
   - NEON 除法使用近似算法，使用适当的 tolerance
   - 浮点数比较使用 `EXPECT_FLOAT_EQ` 而不是 `EXPECT_EQ`

### 调试测试

```bash
# 运行特定测试套件
./bin/test/simd_basic_test --gtest_filter="ArithmeticOperationsTest*"

# 重复运行测试检查稳定性
./bin/test/simd_basic_test --gtest_repeat=100

# 在失败时停止
./bin/test/simd_basic_test --gtest_break_on_failure
```