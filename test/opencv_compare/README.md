# OpenCV对比测试

## 安装OpenCV

### Ubuntu/Debian
```bash
sudo apt-get install libopencv-dev
```

### macOS (Homebrew)
```bash
brew install opencv
```

## 编译对比测试

```bash
# 编译性能对比
g++ -std=c++11 -O3 \
    test/opencv_compare/cv_performance_test.cpp \
    -lopencv_core -lopencv_imgproc \
    -o opencv_benchmark

# 运行对比测试
./opencv_benchmark
```

## 对比项目

1. **图像滤波**
   - Gaussian Blur
   - Box Filter
   - Sobel Edge Detection

2. **颜色空间转换**
   - RGB to Gray
   - RGB to HSV

3. **几何变换**
   - Resize (Bilinear)
   - Rotation
   - Affine Transform

## 性能指标

测试将记录并对比：
- 执行时间
- 内存使用
- 数值精度差异