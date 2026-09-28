// Copyright (C) 2018-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cstdint>
#include <ranges>
#include <span>

namespace dolfinx::la
{
/// Norm types
enum class Norm : std::int8_t
{
  l1,
  l2,
  linf,
  frobenius
};

/// @brief Concept for a contiguous list of process-local indices, e.g.
/// the row or column indices passed to a matrix set function.
template <class R>
concept LocalIndexRange
    = std::ranges::contiguous_range<R> and std::ranges::sized_range<R>
      and std::same_as<std::ranges::range_value_t<R>, std::int32_t>;

/// @brief Matrix accumulate/set concept for functions that can be used
/// in assemblers to accumulate or set values in a matrix.
template <class U, class T>
concept MatSet
    = std::invocable<U, std::span<const std::int32_t>,
                     std::span<const std::int32_t>, std::span<const T>>;
} // namespace dolfinx::la
