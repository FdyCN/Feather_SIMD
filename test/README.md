# GoogleTest集成说明

## 安装GoogleTest

### Ubuntu/Debian
```bash
sudo apt-get install libgtest-dev
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib
```

### macOS (Homebrew)
```bash
brew install googletest
```

### 手动编译
```bash
git clone https://github.com/google/googletest.git
cd googletest
cmake -B build
cmake --build build
```

## 编译测试

```bash
# 编译单元测试
g++ -std=c++11 -isystem /usr/local/include \
    test/unit_tests/*.cpp \
    -lgtest -lgtest_main -pthread \
    -o test_runner

# 运行测试
./test_runner
```

## 测试文件组织

- `simd_test.cpp` - 核心SIMD功能测试
- `math_test.cpp` - 数学库功能测试
- `cv_test.cpp` - CV算子功能测试
- `performance_test.cpp` - 性能基准测试