
#pragma once

#include <array>
#include <cassert>
#include <cstdint>

namespace dolfinx::common
{
/// @brief Return local range, partitioning a global [0, N - 1] range across all
/// ranks into partitions of almost equal size.
/// @param[in] rank The rank of the calling process.
/// @param[in] N The value to partition.
/// @param[in] size Number of ranks across which to partition `N`.
constexpr std::array<std::int64_t, 2> local_range(int rank, std::int64_t N,
                                                  int size)
{
  assert(rank >= 0);
  assert(N >= 0);
  assert(size > 0);

  // Compute number of items per rank and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  if (rank < r)
    return {rank * (n + 1), rank * (n + 1) + n + 1};
  else
    return {rank * n + r, rank * n + r + n};
}
} // namespace dolfinx::common
