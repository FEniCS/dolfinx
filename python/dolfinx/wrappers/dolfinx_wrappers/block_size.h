// Copyright (C) 2026 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace dolfinx_wrappers
{
/// @brief Call `f(bs0, bs1)` with data block sizes `bs`, as
/// `std::integral_constant`s for the sizes with compile-time insertion
/// and as `int`s otherwise.
///
/// Compile-time sizes are (k, k) for k in {1, ..., 9}, and (1, k),
/// (k, 1) for k in {2, 3}.
template <typename F>
void visit_block_size(std::array<int, 2> bs, F&& f)
{
  static constexpr std::array<std::array<int, 2>, 13> sizes{{
      {1, 1},
      {2, 2},
      {3, 3},
      {4, 4},
      {5, 5},
      {6, 6},
      {7, 7},
      {8, 8},
      {9, 9},
      {1, 2},
      {2, 1},
      {1, 3},
      {3, 1},
  }};

  // Call f with sizes[I] as compile-time constants if it equals bs
  auto try_size
      = [&bs, &f]<std::size_t I>(std::integral_constant<std::size_t, I>) -> bool
  {
    if (bs != sizes[I])
      return false;

    f(std::integral_constant<int, sizes[I][0]>{},
      std::integral_constant<int, sizes[I][1]>{});
    return true;
  };

  // Try each entry of sizes in order, stopping at the first match
  bool found = [&try_size]<std::size_t... I>(std::index_sequence<I...>)
  {
    return (try_size(std::integral_constant<std::size_t, I>{}) or ...);
  }(std::make_index_sequence<sizes.size()>{});

  if (!found)
    f(bs[0], bs[1]);
}
} // namespace dolfinx_wrappers
