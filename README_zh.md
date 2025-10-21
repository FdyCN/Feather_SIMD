<div align="center">
  <img src="assets/logo.png" alt="Feather SIMD Logo" width="250"/>

  # Feather SIMD

  **轻量化、高性能、高易用的HEADER-ONLY C++ SIMD 抽象库**

  [English](README.md) | [中文](README_zh.md)

  [![C++11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)](https://en.cppreference.com/w/cpp/11)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![ARM NEON](https://img.shields.io/badge/ARM-NEON-orange.svg)](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
</div>

---

## 概述

Feather SIMD 是一个现代化、模块化的 C++11 SIMD 抽象库，提供自动后端选择的高性能向量运算。该库在前端 API 和后端实现之间实现了清晰的分离，便于扩展和针对不同平台进行优化。

**当前重点：** 开发健壮的 SIMD 头文件库
**未来计划：** 基于此基础构建计算机视觉和深度学习算子

### 架构

库的组织结构采用模块化组件：

- **[core/tiny_simd.hpp](core/tiny_simd.hpp)**：主头文件，包含所有组件
- **[core/base.hpp](core/base.hpp)**：核心 `vec<T, N, Backend>` 接口、类型别名、数学函数和智能后端选择
- **[core/scalar.hpp](core/scalar.hpp)**：标量后端（回退实现）
- **[core/neon.hpp](core/neon.hpp)**：ARM NEON 优化后端

### 核心特性

- **自动后端选择**：基于类型和大小的智能编译期 SIMD 后端选择
- **显式后端控制**：高级用户可以显式指定后端（例如 `vec<float, 4, neon_backend>`）
- **零成本抽象**：后端选择无运行时开销
- **可扩展设计**：易于添加新后端（SSE、AVX 等）
- **类型安全**：强类型检查和编译期保证
- **现代 C++**：要求 C++11 或更高版本

## 数据类型

### 核心向量类

```cpp
template<typename T, size_t N, typename Backend = auto_backend>
class vec;
```

**模板参数：**
- `T`：元素类型（任意算术类型：float、double、fp16_t、int32_t、uint32_t、int16_t、uint16_t、int8_t、uint8_t）
- `N`：向量大小（元素数量，必须 > 0）
- `Backend`：后端类型（默认：`auto_backend` 自动选择）
  - `auto_backend`：自动智能后端选择（推荐）
  - `neon_backend`：显式使用 ARM NEON
  - `scalar_backend`：显式使用标量回退
  - `sse_backend`、`avx_backend`：为未来 x86 支持预留

**成员查询：**
- `vec<T, N, Backend>::is_simd_optimized`：编译期布尔值，指示是否启用 SIMD 优化

### 类型别名

所有类型别名默认使用自动后端选择（`auto_backend`）。在编译时启用 ARM NEON 支持时，相应类型会自动使用 NEON 优化。

#### 浮点向量
- **`vec2f`**：`vec<float, 2>` - 2D 浮点向量（ARM 上 NEON 优化）
- **`vec3f`**：`vec<float, 3>` - 3D 浮点向量（标量）
- **`vec4f`**：`vec<float, 4>` - 4D 浮点向量（ARM 上 NEON 优化）
- **`vec8f`**：`vec<float, 8>` - 8D 浮点向量（标量，支持 AVX 时优化）

#### 双精度浮点向量
- **`vec2d`**：`vec<double, 2>` - 2D 双精度向量（标量）
- **`vec4d`**：`vec<double, 4>` - 4D 双精度向量（标量）

#### 半精度浮点向量
- **`vec4h`**：`vec<fp16_t, 4>` - 4D 半精度向量（标量）
- **`vec8h`**：`vec<fp16_t, 8>` - 8D 半精度向量（ARM 上支持 FP16 时 NEON 优化）
- **`vec16h`**：`vec<fp16_t, 16>` - 16D 半精度向量（标量）

#### 整数向量
- **`vec4i`**：`vec<int32_t, 4>` - 4D int32 向量（ARM 上 NEON 优化）
- **`vec8i`**：`vec<int32_t, 8>` - 8D int32 向量（标量，支持 AVX 时优化）

#### 无符号整数向量
- **`vec4ui`**：`vec<uint32_t, 4>` - 4D uint32 向量（ARM 上 NEON 优化）
- **`vec8ui`**：`vec<uint32_t, 8>` - 8D uint32 向量（标量，支持 AVX 时优化）

#### 短整数向量
- **`vec8s`**：`vec<int16_t, 8>` - 8D int16 向量（标量）
- **`vec16s`**：`vec<int16_t, 16>` - 16D int16 向量（标量）

#### 无符号短整数向量
- **`vec8us`**：`vec<uint16_t, 8>` - 8D uint16 向量（ARM 上 NEON 优化）
- **`vec16us`**：`vec<uint16_t, 16>` - 16D uint16 向量（标量）

#### 字节向量
- **`vec16b`**：`vec<int8_t, 16>` - 16D int8 向量（ARM 上 NEON 优化）
- **`vec32b`**：`vec<int8_t, 32>` - 32D int8 向量（标量）

#### 无符号字节向量
- **`vec16ub`**：`vec<uint8_t, 16>` - 16D uint8 向量（ARM 上 NEON 优化）
- **`vec32ub`**：`vec<uint8_t, 32>` - 32D uint8 向量（标量）

## 后端支持与优化

### 当前后端实现

#### ARM NEON 后端（`neon_backend`）

**优化类型：**
- `float32x2_t`：2 元素浮点向量（`vec2f`）
- `float32x4_t`：4 元素浮点向量（`vec4f`）
- `int32x4_t`：4 元素 int32 向量（`vec4i`）
- `uint32x4_t`：4 元素 uint32 向量（`vec4ui`）
- `uint16x8_t`：8 元素 uint16 向量（`vec8us`）
- `uint8x16_t`：16 元素 uint8 向量（`vec16ub`）
- `int8x16_t`：16 元素 int8 向量（`vec16b`）
- `float16x8_t`：8 元素 fp16 向量（`vec8h`，当定义 `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` 时）

**使用的 NEON 指令：**
- 基础算术：`vaddq_*/vadd_*`、`vsubq_*/vsub_*`、`vmulq_*/vmul_*`
- 除法：`vrecpe_*`、`vrecps_*`（倒数近似 + 牛顿-拉夫逊细化）
- 比较：`vceq_*`
- 最小/最大值：`vminq_*/vmin_*`、`vmaxq_*/vmax_*`
- 绝对值：`vabsq_*/vabs_*`
- 取负：`vnegq_*/vneg_*`
- 加载/存储：`vld1_*`、`vst1_*`、`vdupq_n_*`、`vdup_n_*`

#### 标量后端（`scalar_backend`）

使用 `std::array<T, N>` 的通用 C++ 实现，具有可移植性。在以下情况下使用：
- 平台不支持 SIMD
- 向量大小与 SIMD 寄存器大小不匹配
- 显式请求标量后端

### 智能后端选择

库在编译期自动选择最优后端，基于：
1. **平台能力**：通过 `TINY_SIMD_ARM_NEON`、`TINY_SIMD_X86_SSE` 等检测
2. **类型和大小匹配**：仅当 `(T, N)` 匹配 NEON 寄存器类型时选择 NEON 后端
3. **性能优势**：仅在提供可衡量优势时选择后端

**NEON 选择规则：**
```cpp
// float: vec2f 和 vec4f 使用 NEON
vec<float, 2> -> neon_backend  // float32x2_t
vec<float, 4> -> neon_backend  // float32x4_t
vec<float, 3> -> scalar_backend // 无匹配的 NEON 类型

// int32/uint32: vec4i 和 vec4ui 使用 NEON
vec<int32_t, 4>  -> neon_backend  // int32x4_t
vec<uint32_t, 4> -> neon_backend  // uint32x4_t

// int16/uint16: vec8us 使用 NEON
vec<uint16_t, 8> -> neon_backend  // uint16x8_t
vec<int16_t, 8>  -> scalar_backend // NEON 支持尚未实现

// int8/uint8: vec16b 和 vec16ub 使用 NEON
vec<int8_t, 16>  -> neon_backend  // int8x16_t
vec<uint8_t, 16> -> neon_backend  // uint8x16_t

// fp16: vec8h 使用 NEON（如果支持 FP16 算术）
vec<fp16_t, 8> -> neon_backend  // float16x8_t（有条件）
```

### 未来后端支持

计划中的后端实现：
- **`sse_backend`**：x86 SSE 支持（128 位）
- **`avx_backend`**：x86 AVX 支持（256 位）
- **`avx2_backend`**：x86 AVX2 支持，增强整数运算

## API 参考

### 构造函数

```cpp
vec()                                    // 默认：零初始化
explicit vec(T scalar)                   // 将标量广播到所有元素
vec(std::initializer_list<T> init)      // 从列表初始化：{1, 2, 3, 4}
vec(const T* ptr)                        // 从内存加载（非对齐）
static vec load_aligned(const T* ptr)    // 从对齐内存加载
```

### 数据访问

```cpp
T operator[](size_t i) const            // 元素访问（只读）
T* data()                                // 获取可变数据指针
const T* data() const                    // 获取常量数据指针
size_t size() const                      // 获取向量大小（返回 N）
void store(T* ptr) const                 // 存储到内存（非对齐）
void store_aligned(T* ptr) const         // 存储到对齐内存
```

### 算术运算

所有算术运算支持向量-向量和向量-标量两种变体：

```cpp
vec operator+(const vec& other) const    // 加法
vec operator-(const vec& other) const    // 减法
vec operator*(const vec& other) const    // 逐元素乘法
vec operator/(const vec& other) const    // 逐元素除法
vec operator-() const                    // 一元取负

vec& operator+=(const vec& other)        // 原地加法
vec& operator-=(const vec& other)        // 原地减法
vec& operator*=(const vec& other)        // 原地乘法
vec& operator/=(const vec& other)        // 原地除法

// 标量变体
vec operator+(T scalar) const            // 所有元素加标量
vec operator-(T scalar) const            // 所有元素减标量
vec operator*(T scalar) const            // 所有元素乘标量
vec operator/(T scalar) const            // 所有元素除标量
```

### 比较运算

```cpp
bool operator==(const vec& other) const  // 逐元素相等检查
bool operator!=(const vec& other) const  // 逐元素不等检查
```

### 数学函数

所有数学函数都是 `tiny_simd` 命名空间中的自由函数。

#### 基础向量运算

```cpp
T dot(const vec<T, N>& a, const vec<T, N>& b)          // 点积
T length(const vec<T, N>& v)                            // 向量长度
T length_squared(const vec<T, N>& v)                    // 长度平方（避免 sqrt）
vec<T, N> normalize(const vec<T, N>& v)                 // 单位向量
T distance(const vec<T, N>& a, const vec<T, N>& b)      // 点间距离
T distance_squared(const vec<T, N>& a, const vec<T, N>& b) // 距离平方
```

#### 高级向量运算

```cpp
vec<T, N> lerp(const vec<T, N>& a, const vec<T, N>& b, T t)  // 线性插值
vec<T, N> project(const vec<T, N>& a, const vec<T, N>& b)    // 将 a 投影到 b
vec<T, N> reflect(const vec<T, N>& v, const vec<T, N>& n)    // 沿法线 n 反射 v
vec<T, 3> cross(const vec<T, 3>& a, const vec<T, 3>& b)      // 叉积（仅 3D）
```

#### 逐元素函数

```cpp
vec<T, N> min(const vec<T, N>& a, const vec<T, N>& b)   // 逐元素最小值
vec<T, N> max(const vec<T, N>& a, const vec<T, N>& b)   // 逐元素最大值
vec<T, N> clamp(const vec<T, N>& v, T min_val, T max_val) // 限制到范围
vec<T, N> clamp(const vec<T, N>& v, const vec<T, N>& min_val, const vec<T, N>& max_val)
vec<T, N> abs(const vec<T, N>& v)                       // 逐元素绝对值
```

#### 防溢出算术

这些函数通过使用更宽的数据类型或饱和来防止溢出：

**拓宽运算**（返回更大类型以防止溢出）：
```cpp
// uint8 运算 -> 返回 uint16
vec<uint16_t, N> add_wide(const vec<uint8_t, N>& a, const vec<uint8_t, N>& b)
vec<uint16_t, N> mul_wide(const vec<uint8_t, N>& a, const vec<uint8_t, N>& b)

// uint16 运算 -> 返回 uint32
vec<uint32_t, N> add_wide(const vec<uint16_t, N>& a, const vec<uint16_t, N>& b)
vec<uint32_t, N> mul_wide(const vec<uint16_t, N>& a, const vec<uint16_t, N>& b)

// int8 运算 -> 返回 int16
vec<int16_t, N> add_wide(const vec<int8_t, N>& a, const vec<int8_t, N>& b)
vec<int16_t, N> mul_wide(const vec<int8_t, N>& a, const vec<int8_t, N>& b)
```

**饱和运算**（限制到类型边界）：
```cpp
// uint8 饱和运算（限制到 [0, 255]）
vec<uint8_t, N> add_sat(const vec<uint8_t, N>& a, const vec<uint8_t, N>& b)
vec<uint8_t, N> sub_sat(const vec<uint8_t, N>& a, const vec<uint8_t, N>& b)

// uint16 饱和运算（限制到 [0, 65535]）
vec<uint16_t, N> add_sat(const vec<uint16_t, N>& a, const vec<uint16_t, N>& b)
vec<uint16_t, N> sub_sat(const vec<uint16_t, N>& a, const vec<uint16_t, N>& b)

// int8 饱和运算（限制到 [-128, 127]）
vec<int8_t, N> add_sat(const vec<int8_t, N>& a, const vec<int8_t, N>& b)
vec<int8_t, N> sub_sat(const vec<int8_t, N>& a, const vec<int8_t, N>& b)
```

## 配置

### 编译期标志

定义这些宏以启用特定的 SIMD 后端：

- **`TINY_SIMD_ARM_NEON`**：启用 ARM NEON 优化
- **`TINY_SIMD_X86_SSE`**：启用 x86 SSE 优化（计划中）
- **`TINY_SIMD_X86_AVX`**：启用 x86 AVX 优化（计划中）
- **`TINY_SIMD_X86_AVX2`**：启用 x86 AVX2 优化（计划中）

**CMake 配置示例：**
```cmake
# 在 ARM 平台上启用 NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    target_compile_definitions(my_target PRIVATE TINY_SIMD_ARM_NEON)
endif()
```

### 运行时配置

通过 `tiny_simd::config` 命名空间查询平台能力和限制：

```cpp
namespace tiny_simd::config {
    // 平台检测
    constexpr bool has_neon;        // ARM NEON 可用
    constexpr bool has_sse;         // x86 SSE 可用
    constexpr bool has_avx;         // x86 AVX 可用
    constexpr bool has_avx2;        // x86 AVX2 可用

    // 每种类型的最大向量大小
    constexpr size_t max_vector_size_float;
    constexpr size_t max_vector_size_double;
    constexpr size_t max_vector_size_fp16;
    constexpr size_t max_vector_size_int32;
    constexpr size_t max_vector_size_uint32;
    constexpr size_t max_vector_size_int16;
    constexpr size_t max_vector_size_uint16;
    constexpr size_t max_vector_size_int8;
    constexpr size_t max_vector_size_uint8;

    // 内存对齐要求
    constexpr size_t simd_alignment;  // NEON/SSE 为 16，AVX 为 32
}
```

## 使用示例

### 基础用法

```cpp
#include "core/tiny_simd.hpp"
using namespace tiny_simd;

// 创建向量（自动后端选择）
vec4f a{1.0f, 2.0f, 3.0f, 4.0f};        // ARM 上使用 NEON
vec4f b{5.0f, 6.0f, 7.0f, 8.0f};

// 基础算术（支持时 SIMD 优化）
vec4f sum = a + b;                       // {6, 8, 10, 12}
vec4f product = a * 2.0f;                // {2, 4, 6, 8}
vec4f negated = -a;                      // {-1, -2, -3, -4}

// 向量运算
float dp = dot(a, b);                    // 点积：70
float len = length(a);                   // 长度：~5.477
vec4f normalized = normalize(a);         // 单位向量

// 元素访问
float x = a[0];                          // 1.0f
float y = a[1];                          // 2.0f
```

### 使用不同类型

```cpp
// 整数向量
vec4i int_a{1, 2, 3, 4};                 // ARM 上使用 NEON
vec4i int_b{5, 6, 7, 8};
vec4i int_sum = int_a + int_b;           // {6, 8, 10, 12}

// 无符号整数向量
vec4ui uint_a{1, 2, 3, 4};               // ARM 上使用 NEON
vec4ui uint_b{5, 6, 7, 8};

// 图像处理的字节向量
vec16ub pixels{10, 20, 30, 40, /*...*/}; // ARM 上使用 NEON
vec16ub brightened = pixels + vec16ub(50);

// 机器学习/深度神经网络的半精度
vec8h half_weights{1.0f, 2.0f, /*...*/}; // 如果支持 FP16 则使用 NEON
```

### 防溢出算术

```cpp
// 示例：带溢出保护的图像处理
vec16ub pixel_a{200, 150, 255, 100, /*...*/};
vec16ub pixel_b{100, 200, 50, 50, /*...*/};

// 常规加法（可能溢出/环绕）
vec16ub regular_sum = pixel_a + pixel_b;
// 结果：{44, 94, 49, 150, ...}（200+100=44 由于环绕）

// 饱和加法（限制到 [0, 255]）
vec16ub sat_sum = add_sat(pixel_a, pixel_b);
// 结果：{255, 255, 255, 150, ...}（限制到最大值 255）

// 拓宽加法（返回 uint16 以保存完整结果）
auto wide_sum = add_wide(pixel_a, pixel_b);
// 结果：{300, 350, 305, 150, ...}（无溢出，返回 vec<uint16_t, 16>）

// 精度乘法的拓宽
vec16ub scale_a{10, 20, 30, 40, /*...*/};
vec16ub scale_b{10, 15, 20, 25, /*...*/};
auto wide_product = mul_wide(scale_a, scale_b);
// 结果：{100, 300, 600, 1000, ...}（返回 vec<uint16_t, 16>）
```

### 显式后端选择

```cpp
// 显式使用 NEON 后端
vec<float, 4, neon_backend> neon_vec{1.0f, 2.0f, 3.0f, 4.0f};

// 显式使用标量后端
vec<float, 4, scalar_backend> scalar_vec{1.0f, 2.0f, 3.0f, 4.0f};

// 检查是否使用 SIMD
static_assert(vec4f::is_simd_optimized, "vec4f 应在 ARM 上使用 SIMD");
```

### 高级：自定义向量大小

```cpp
// 支持任意大小（自动选择最优后端）
vec<float, 7> custom_vec{1, 2, 3, 4, 5, 6, 7};  // 使用标量后端
vec<float, 16> large_vec;                        // 使用标量后端

// 查询选择的后端
using backend_type = vec<float, 7>::backend_type;
constexpr bool is_optimized = vec<float, 7>::is_simd_optimized;  // false
```

### 内存操作

```cpp
// 从对齐内存加载
alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
vec4f loaded = vec4f::load_aligned(data);

// 存储到内存
float output[4];
loaded.store(output);

// 对齐存储以获得最佳性能
alignas(16) float aligned_output[4];
loaded.store_aligned(aligned_output);
```

## 性能特征

### SIMD 性能提升

当 SIMD 后端可用时，预期显著的性能提升：

| 操作 | 标量 | NEON (ARM) | SSE (x86) | 加速比 |
|------|------|------------|-----------|--------|
| vec4f 加法 | 1.0x | 4.0x | 4.0x | 2-4x |
| vec4f 乘法 | 1.0x | 4.0x | 4.0x | 2-4x |
| vec4f 点积 | 1.0x | 4.0x | 4.0x | 3-5x |
| vec16ub 运算 | 1.0x | 8-16x | 8-16x | 4-8x |

*注意：实际加速比取决于工作负载、内存访问模式和编译器优化。*

### 后端选择性能

- **零运行时开销**：后端选择在编译期进行
- **无虚函数**：所有操作都是静态分发
- **内联友好**：大多数操作完全内联
- **缓存高效**：使用 `load_aligned()` / `store_aligned()` 时对齐内存访问

### 最佳实践

1. **尽可能使用对齐内存**：NEON/SSE 使用 `alignas(16)`，AVX 使用 `alignas(32)`
   ```cpp
   alignas(16) float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
   vec4f v = vec4f::load_aligned(data);
   ```

2. **优先使用类型别名**：使用 `vec4f` 而不是 `vec<float, 4>` 以提高可读性
   ```cpp
   vec4f v;  // 清晰简洁
   ```

3. **对窄类型的整数运算使用防溢出操作**：
   ```cpp
   // 图像处理时使用饱和算术
   vec16ub result = add_sat(pixels, brightness);

   // 精度关键工作时使用拓宽算术
   auto precise_result = mul_wide(a, b);
   ```

4. **利用编译期查询**优化代码路径：
   ```cpp
   if constexpr (vec4f::is_simd_optimized) {
       // SIMD 特定优化
   }
   ```

5. **最小化存储/加载操作**：尽可能长时间将数据保持在向量中
   ```cpp
   // 好：在向量中链式操作
   vec4f result = normalize(a + b * 2.0f);

   // 效率较低：存储和重新加载
   vec4f temp = a + b;
   float data[4];
   temp.store(data);
   vec4f result = vec4f(data) * 2.0f;
   ```

### 溢出处理性能

不同的溢出策略具有不同的性能特征：

| 策略 | 性能 | 安全性 | 使用场景 |
|------|------|--------|----------|
| 常规算术（`+`、`*`） | 最快 | 无（环绕） | 不可能溢出或可接受时 |
| 饱和（`add_sat`、`sub_sat`） | 中等（~1.2-2x 开销） | 限制到边界 | 图像处理、音频处理 |
| 拓宽（`add_wide`、`mul_wide`） | 低开销 | 完全（返回更大类型） | 精度关键计算 |

**NEON 优化状态：**
- 饱和运算：通过 `vqadd_*`、`vqsub_*` intrinsics 硬件加速
- 拓宽运算：通过 `vaddl_*`、`vmull_*` intrinsics 硬件加速

## 测试

库包含全面的单元测试以确保正确性：

- **[test/unit_tests/simd_basic_test.cpp](../test/unit_tests/simd_basic_test.cpp)**：基础 SIMD 操作测试
- **[test/unit_tests/vec2f_neon_test.cpp](../test/unit_tests/vec2f_neon_test.cpp)**：NEON 特定 vec2f 测试
- **[test/unit_tests/overflow_test.cpp](../test/unit_tests/overflow_test.cpp)**：防溢出算术测试
- **[test/unit_tests/opencv_comparison_test.cpp](../test/unit_tests/opencv_comparison_test.cpp)**：与 OpenCV 的精度和性能对比测试（可选）

### 运行测试

基础测试（无依赖）：
```bash
mkdir build && cd build
cmake ..
make
ctest
# 或直接运行
./bin/test/tiny_simd_unit_tests
```

### 使用 OpenCV 进行测试（可选）

为了进行精度验证和与 OpenCV 的性能基准测试，你可以启用 OpenCV 支持：

#### 安装 OpenCV

**macOS:**
```bash
brew install opencv
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libopencv-dev
```

**其他 Linux:**
```bash
# 使用你的发行版的包管理器
# 例如：yum install opencv-devel (Fedora/RHEL)
```

#### 使用 OpenCV 构建

```bash
mkdir build && cd build
cmake -DTINY_SIMD_WITH_OPENCV=ON ..
make
./bin/test/tiny_simd_unit_tests --gtest_filter="OpenCVComparison.*"
```

OpenCV 对比测试将会：
- 针对 OpenCV 的参考实现验证 SIMD 实现的精度
- 测量 SIMD 和 OpenCV 操作之间的性能差异
- 报告最大误差和加速比指标

**注意：** OpenCV 是完全可选的。核心库没有依赖，在没有 OpenCV 的情况下也能完美工作。OpenCV 测试仅用于验证和基准测试目的。

## 需求

- **C++ 标准**：C++11 或更高版本
- **编译器支持**：
  - GCC 4.8+
  - Clang 3.4+
  - Apple Clang（Xcode）
  - MSVC 2015+（用于未来 x86 支持）
- **平台支持**：
  - ARM（32 位和 64 位）带 NEON
  - x86/x86-64（计划中的 SSE/AVX 支持）
  - 任何平台（标量回退）

## 许可证

此库是 Feather 项目的一部分。

## 贡献

添加新后端或优化时：

1. 创建新的后端文件（例如 `core/sse.hpp`）
2. 为您的后端特化 `backend_ops<backend_tag, T, N>`
3. 更新 [base.hpp](base.hpp) 中的 `neon_has_advantage`（或创建类似 trait）
4. 在配置中添加编译期标志
5. 添加全面的测试
6. 更新此 README

## 参见

- [项目根目录 README](../README.md)
- [示例和基准测试](../test/benchmarks/)
- [单元测试](../test/unit_tests/)
