// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "sfc.h"
#include "AdjacencyList.h"
#include "partition.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/sort.h>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

using namespace dolfinx;

namespace
{
/// Number of bits per coordinate used to quantise positions before a
/// space-filling curve key is computed. With three dimensions this gives
/// a 63-bit key.
constexpr int nbits = 21;

/// @brief Interleave the low `nbits` bits of up to three coordinates into
/// a Morton ('Z-order') curve key.
std::uint64_t morton_key(std::array<std::uint32_t, 3> c, int /*gdim*/)
{
  // Spread the low bits of `v` out so that they occupy every third bit
  // position
  auto spread = [](std::uint64_t v) -> std::uint64_t
  {
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffff;
    v = (v | v << 16) & 0x1f0000ff0000ff;
    v = (v | v << 8) & 0x100f00f00f00f00f;
    v = (v | v << 4) & 0x10c30c30c30c30c3;
    v = (v | v << 2) & 0x1249249249249249;
    return v;
  };

  return spread(c[0]) | (spread(c[1]) << 1) | (spread(c[2]) << 2);
}

/// @brief Distance along a Hilbert curve of a point with quantised
/// coordinates.
///
/// Uses Skilling's algorithm to transform the coordinates in place into
/// the 'transpose' of the Hilbert index (J. Skilling, Programming the
/// Hilbert curve, AIP Conf. Proc. 707, 2004), then interleaves the
/// transpose to give the index itself.
///
/// @param[in] c Quantised coordinates, each using the low `nbits` bits.
/// @param[in] gdim Number of coordinate components. The curve is
/// constructed in `gdim` dimensions, so that (unlike a Morton key) the
/// unused components cannot break the ordering.
/// @return Distance along the curve, using `gdim * nbits` bits.
std::uint64_t hilbert_key(std::array<std::uint32_t, 3> c, int gdim)
{
  // Transform the coordinates to the Hilbert index transpose
  const std::uint32_t m = std::uint32_t(1) << (nbits - 1);
  for (std::uint32_t q = m; q > 1; q >>= 1)
  {
    const std::uint32_t p = q - 1;
    for (int i = 0; i < gdim; ++i)
    {
      if (c[i] & q)
        c[0] ^= p; // Invert
      else
      {
        // Exchange
        const std::uint32_t t = (c[0] ^ c[i]) & p;
        c[0] ^= t;
        c[i] ^= t;
      }
    }
  }

  // Gray encode
  for (int i = 1; i < gdim; ++i)
    c[i] ^= c[i - 1];
  std::uint32_t t = 0;
  for (std::uint32_t q = m; q > 1; q >>= 1)
  {
    if (c[gdim - 1] & q)
      t ^= q - 1;
  }
  for (int i = 0; i < gdim; ++i)
    c[i] ^= t;

  // Interleave the transpose, most significant bit of c[0] first
  std::uint64_t key = 0;
  for (int j = nbits - 1; j >= 0; --j)
  {
    for (int i = 0; i < gdim; ++i)
      key = (key << 1) | ((c[i] >> j) & 1);
  }

  return key;
}

/// @brief Partition points into `nparts` groups of (approximately) equal
/// size by their position along a space-filling curve.
///
/// @param[in] comm MPI communicator the points are distributed across.
/// @param[in] nparts Number of partitions.
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point.
/// @param[in] key Curve key for a point, given its quantised coordinates
/// and `gdim`.
/// @return Partition index in `[0, nparts)` for each point.
template <typename K>
std::vector<int> partition_by_curve(MPI_Comm comm, int nparts,
                                    std::span<const double> x, int gdim, K key)
{
  if (gdim < 1 or gdim > 3)
    throw std::runtime_error("Geometric dimension must be 1, 2 or 3.");
  if (nparts < 1)
    throw std::runtime_error("Number of partitions must be > 0.");
  if (x.size() % gdim != 0)
  {
    throw std::runtime_error(
        "Point coordinate array size is not a multiple of gdim.");
  }

  const std::size_t num_points = x.size() / gdim;
  if (nparts == 1)
    return std::vector<int>(num_points, 0);

  // Global bounding box of the points. Note: the reduction is over
  // {-min, max} so that a single MPI_MAX reduction suffices.
  std::array<double, 6> extent;
  extent.fill(std::numeric_limits<double>::lowest());
  for (std::size_t i = 0; i < num_points; ++i)
  {
    for (int d = 0; d < gdim; ++d)
    {
      extent[d] = std::max(extent[d], -x[gdim * i + d]);
      extent[3 + d] = std::max(extent[3 + d], x[gdim * i + d]);
    }
  }
  {
    std::array<double, 6> recv;
    MPI_Allreduce(extent.data(), recv.data(), 6, MPI_DOUBLE, MPI_MAX, comm);
    extent = recv;
  }

  // Curve key for each point, from its position in the bounding box
  // scaled to the bit range of the key
  constexpr double range = (1 << nbits) - 1;
  std::array<double, 3> scale = {0, 0, 0};
  for (int d = 0; d < gdim; ++d)
  {
    const double width = extent[3 + d] + extent[d];
    scale[d] = width > 0 ? range / width : 0;
  }

  std::vector<std::uint64_t> keys(num_points);
  for (std::size_t i = 0; i < num_points; ++i)
  {
    std::array<std::uint32_t, 3> c = {0, 0, 0};
    for (int d = 0; d < gdim; ++d)
    {
      c[d] = static_cast<std::uint32_t>(scale[d]
                                        * (x[gdim * i + d] + extent[d]));
    }
    keys[i] = key(c, gdim);
  }

  // Sample the local keys, over-sampling by a fixed factor per
  // partition, and gather the samples on all ranks. Splitters taken at
  // equal-count positions of the sorted sample then cut the global key
  // order into `nparts` pieces of approximately equal size, without
  // sorting the keys across ranks.
  //
  // Note: the local keys are sorted first so that the sample is a
  // systematic sample of the local quantiles. Sampling the keys in their
  // incoming order instead makes the sample a random one, whose much
  // larger variance shows up directly as partition imbalance (measured:
  // 7% versus under 1% for a mesh whose cells arrive in an order
  // unrelated to their position).
  std::vector<std::uint64_t> sample;
  {
    constexpr int oversample = 32;
    std::vector<std::uint64_t> keys_sorted(keys);
    dolfinx::radix_sort(keys_sorted);
    const std::size_t n = std::min(
        keys_sorted.size(), static_cast<std::size_t>(oversample) * nparts);
    sample.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      sample[i] = keys_sorted[(i * keys_sorted.size()) / n];
  }

  std::vector<std::uint64_t> sample_all;
  {
    const int size = dolfinx::MPI::size(comm);
    int num_local = sample.size();
    std::vector<int> counts(size), displs(size + 1, 0);
    MPI_Allgather(&num_local, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::partial_sum(counts.begin(), counts.end(), std::next(displs.begin()));
    sample_all.resize(displs.back());
    MPI_Allgatherv(sample.data(), num_local, MPI_UINT64_T, sample_all.data(),
                   counts.data(), displs.data(), MPI_UINT64_T, comm);
    dolfinx::radix_sort(sample_all);
  }

  // Degenerate case: no points anywhere
  if (sample_all.empty())
    return std::vector<int>(num_points, 0);

  std::vector<std::uint64_t> splitters(nparts - 1);
  for (int p = 1; p < nparts; ++p)
    splitters[p - 1] = sample_all[(sample_all.size() * p) / nparts];

  std::vector<int> part(num_points);
  std::ranges::transform(keys, part.begin(),
                         [&splitters](std::uint64_t k)
                         {
                           return std::ranges::distance(
                               splitters.begin(),
                               std::ranges::upper_bound(splitters, k));
                         });

  return part;
}
} // namespace

//-----------------------------------------------------------------------------
std::vector<int> graph::partition_sfc_morton(MPI_Comm comm, int nparts,
                                             std::span<const double> x,
                                             int gdim)
{
  common::Timer timer("Compute Morton SFC partition of points");
  return partition_by_curve(comm, nparts, x, gdim, morton_key);
}
//-----------------------------------------------------------------------------
std::vector<int> graph::partition_sfc_hilbert(MPI_Comm comm, int nparts,
                                              std::span<const double> x,
                                              int gdim)
{
  common::Timer timer("Compute Hilbert SFC partition of points");
  return partition_by_curve(comm, nparts, x, gdim, hilbert_key);
}
//-----------------------------------------------------------------------------
