// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "local_range.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace dolfinx::common
{
/// @brief Split `[0, N)` into contiguous chunks and apply `f` to each,
/// in parallel.
///
/// Chunk 0 runs on the calling thread and the call returns once every
/// chunk is complete. The number of chunks is capped at `N`, so a
/// chunk is never empty.
///
/// @param[in] N Size of the range to split.
/// @param[in] num_threads Maximum number of threads to use. Must be
/// >= 1.
/// @param[in] f Invoked as `f(chunk, i0, i1)` for the chunk covering
/// `[i0, i1)`. `chunk` indexes any per-chunk output buffers, which
/// must therefore be sized for `num_threads` chunks.
/// @throws std::invalid_argument If `num_threads < 1`.
template <typename F>
void parallel_for(std::int64_t N, int num_threads, F f)
{
  if (num_threads < 1)
    throw std::invalid_argument("num_threads must be >= 1.");

  int nchunks = static_cast<int>(
      std::max<std::int64_t>(1, std::min<std::int64_t>(num_threads, N)));
  std::vector<std::jthread> threads;
  threads.reserve(nchunks - 1);
  for (int i = 1; i < nchunks; ++i)
  {
    std::array<std::int64_t, 2> r = local_range(i, N, nchunks);
    threads.emplace_back(
        [&f, i, r]
        {
          f(i, static_cast<std::size_t>(r[0]), static_cast<std::size_t>(r[1]));
        });
  }

  std::array<std::int64_t, 2> r = local_range(0, N, nchunks);
  f(0, static_cast<std::size_t>(r[0]), static_cast<std::size_t>(r[1]));
}
} // namespace dolfinx::common
