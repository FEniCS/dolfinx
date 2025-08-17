// Copyright (C) 2024 Joseph P. Dean and Garth N. Wells
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cstdint>
#include <span>
#include <type_traits>

#include <dolfinx/common/types.h>

#include <basix/mdspan.hpp>

namespace dolfinx::fem
{

/// @brief DOF transform kernel concept.
template <class U, class T>
concept DofTransformKernel
    = std::is_invocable_v<U, std::span<T>, std::span<const std::uint32_t>,
                          std::int32_t, int>;

/// @brief Finite element cell kernel concept.
///
/// Kernel functions that can be passed to an assembler for execution
/// must satisfy this concept.
template <class U, class T>
concept FEkernel
    = std::is_invocable_v<U, T*, const T*, const T*, const scalar_value_t<T>*,
                          const int*, const std::uint8_t*, void*>;

/// @brief Concept for mdspan of rank 1 or 2.
template <class T>
concept MDSpan2
    = std::is_convertible_v<
          std::remove_cvref_t<T>,
          md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>>
      or std::is_convertible_v<
          std::remove_cvref_t<T>,
          md::mdspan<const std::int32_t, md::dextents<std::size_t, 1>>>;

} // namespace dolfinx::fem
