/*
 * Copyright (c) 2025 FdyCN
 *
 * Distributed under MIT license.
 * See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
 */

#ifndef TINY_SIMD_HPP
#define TINY_SIMD_HPP

//=============================================================================
// Tiny SIMD - Modular SIMD Abstraction Library
//
// Architecture:
//   core/base.hpp   - Core interfaces and smart backend dispatching
//   core/scalar.hpp - Scalar backend (fallback)
//   core/neon.hpp   - ARM NEON backend
//
// Usage:
//   #include "tiny_simd.hpp"
//
//   using namespace tiny_simd;
//
//   // Automatic backend selection
//   vec<float, 4> v1({1.0f, 2.0f, 3.0f, 4.0f});
//   vec<float, 4> v2({5.0f, 6.0f, 7.0f, 8.0f});
//   auto result = v1 + v2;
//
//   // Or use type aliases
//   vec4f positions({1.0f, 2.0f, 3.0f, 4.0f});
//
//   // Explicit backend selection (advanced)
//   vec<float, 4, neon_backend> explicit_neon;
//   vec<float, 4, scalar_backend> explicit_scalar;
//
//=============================================================================

// Include modular components
#include "base.hpp"
#include "scalar.hpp"
#include "neon.hpp"
#include "avx2.hpp"

#endif // TINY_SIMD_HPP
