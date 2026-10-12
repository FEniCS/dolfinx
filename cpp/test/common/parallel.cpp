// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for common::parallel_for. The threaded mesh code relies
// on the chunk decomposition being a partition of [0, N) in increasing
// chunk order, and on the chunk count being capped at N, so that
// per-chunk buffers sized for `num_threads` can be concatenated in
// chunk order.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/parallel.h>
#include <mutex>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

TEST_CASE("parallel_for covers the range exactly once", "[parallel_for]")
{
  for (std::int64_t N : {std::int64_t(0), std::int64_t(1), std::int64_t(2),
                         std::int64_t(17), std::int64_t(1000)})
  {
    for (int num_threads : {1, 2, 3, 8, 64})
    {
      std::vector<std::int8_t> visits(N, 0);
      std::mutex mutex;
      std::vector<std::array<std::size_t, 2>> chunks;
      common::parallel_for(
          N, num_threads,
          [&visits, &chunks, &mutex](int c, std::size_t i0, std::size_t i1)
          {
            for (std::size_t i = i0; i < i1; ++i)
              ++visits[i];
            std::scoped_lock lock(mutex);
            chunks.push_back({static_cast<std::size_t>(c), i0});
            CHECK(i0 <= i1);
          });

      // Every element visited exactly once
      CHECK(std::ranges::all_of(visits, [](std::int8_t v) { return v == 1; }));

      // Chunk count is capped at N, so no chunk is empty
      CHECK(chunks.size()
            <= static_cast<std::size_t>(std::max<std::int64_t>(1, N)));

      // Chunk indices are 0..nchunks-1, and increase with the range
      std::ranges::sort(chunks);
      for (std::size_t c = 0; c < chunks.size(); ++c)
        CHECK(chunks[c][0] == c);
      CHECK(std::ranges::is_sorted(chunks, {}, [](std::array<std::size_t, 2> c)
                                   { return c[1]; }));
    }
  }
}

TEST_CASE("parallel_for rejects num_threads < 1", "[parallel_for]")
{
  auto noop = [](int, std::size_t, std::size_t) {};
  CHECK_THROWS_AS(common::parallel_for(10, 0, noop), std::invalid_argument);
  CHECK_THROWS_AS(common::parallel_for(10, -1, noop), std::invalid_argument);
}
