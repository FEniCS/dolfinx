
// Copyright (C) 2007-2023 Magnus Vikstrøm, Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cassert>
#include <cstdint>

namespace dolfinx::common
{
/// @brief Partition a global range [0, N - 1] across callers into
/// non-overlapping sub-partitions of almost equal size. Returns the
/// local partition for the caller. The local partition range.
///
/// Partitions [0, N) into `size` non-overlapping partitions `[n_(i0},
/// n_(i1))`, where `i` is `index` and `n_(i1) == n_((i+1)0)`.
///
/// @param[in] index Index of the partition to compute.
/// @param[in] N Global range to partition.
/// @param[in] size Number of partitions into which to partition `N`.
constexpr std::array<std::int64_t, 2> local_range(int index, std::int64_t N,
                                                  int size)
{
  assert(index >= 0);
  assert(N >= 0);
  assert(size > 0);

  // Compute number of items per rank and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  if (index < r)
    return {index * (n + 1), index * (n + 1) + n + 1};
  else
    return {index * n + r, index * n + r + n};
}
} // namespace dolfinx::common
