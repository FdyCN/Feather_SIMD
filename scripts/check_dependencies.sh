#!/bin/bash

# 检查项目依赖版本兼容性

set -e

echo "=== 检查依赖版本兼容性 ==="

# 检查 GoogleTest 版本
if [ -d "third_party/googletest" ]; then
    cd third_party/googletest

    # 尝试多种方式获取版本
    GTEST_VERSION=$(git describe --tags --exact-match 2>/dev/null || \
                   git describe --tags 2>/dev/null || \
                   git rev-parse --short HEAD)

    echo "GoogleTest 版本: $GTEST_VERSION"

    # 检查是否是兼容版本
    if [[ "$GTEST_VERSION" =~ release-1\.12\. ]] || [[ "$GTEST_VERSION" == "v1.12.0" ]]; then
        echo "✅ GoogleTest 版本兼容 C++11"
    elif [[ "$GTEST_VERSION" =~ release-1\.1[0-1]\. ]] || [[ "$GTEST_VERSION" =~ ^v1\.1[0-1]\. ]]; then
        echo "✅ GoogleTest 版本兼容 C++11（较旧版本）"
    elif [[ "$GTEST_VERSION" =~ release-1\.13\. ]] || [[ "$GTEST_VERSION" =~ ^v1\.13\. ]]; then
        echo "⚠️  警告: GoogleTest v1.13.x 要求 C++14"
    elif [[ "$GTEST_VERSION" =~ release-1\.1[4-9]\. ]] || [[ "$GTEST_VERSION" =~ ^v1\.1[4-9]\. ]]; then
        echo "❌ 错误: GoogleTest $GTEST_VERSION 要求 C++17，与项目的 C++11 不兼容"
        exit 1
    elif [[ "$GTEST_VERSION" =~ release-1\.[2-9][0-9]\. ]] || [[ "$GTEST_VERSION" =~ ^v1\.[2-9][0-9]\. ]]; then
        echo "❌ 错误: GoogleTest $GTEST_VERSION 要求 C++17，与项目的 C++11 不兼容"
        exit 1
    else
        echo "⚠️  警告: 未知的 GoogleTest 版本 $GTEST_VERSION，请手动验证兼容性"
    fi
    cd ../..
else
    echo "❌ 错误: GoogleTest submodule 未找到"
    echo "请运行: git submodule update --init --recursive"
    exit 1
fi

# 检查 CMake C++ 标准设置
if [ -f "CMakeLists.txt" ]; then
    CPP_STANDARD=$(grep "CMAKE_CXX_STANDARD" CMakeLists.txt | head -1 | grep -o '[0-9]\+' || echo "未设置")
    echo "项目 C++ 标准: $CPP_STANDARD"

    if [ "$CPP_STANDARD" = "11" ]; then
        echo "✅ C++ 标准设置正确"
    else
        echo "⚠️  警告: C++ 标准不是 11，可能影响兼容性"
    fi
fi

echo ""
echo "💡 提示:"
echo "  - 要升级 GoogleTest: 需先升级项目到 C++17"
echo "  - 要保持 C++11: 继续使用 GoogleTest v1.12.0"
echo "  - 验证构建: 运行 'cmake --build build --target test'"
echo ""
echo "=== 检查完成 ==="