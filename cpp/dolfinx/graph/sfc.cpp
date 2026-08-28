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
#include <optional>
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
/// weight by their position along a space-filling curve.
///
/// @param[in] comm MPI communicator the points are distributed across.
/// @param[in] nparts Number of partitions.
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point.
/// @param[in] key Curve key for a point, given its quantised coordinates
/// and `gdim`.
/// @param[in] weights Point weights, one entry per row of `x`. If
/// `std::nullopt`, points are treated as having equal weight.
/// @return Partition index in `[0, nparts)` for each point.
template <typename K>
std::vector<int>
partition_by_curve(MPI_Comm comm, int nparts, std::span<const double> x,
                   int gdim, K key,
                   std::optional<std::span<const std::int32_t>> weights)
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
  if (weights and weights->size() != num_points)
  {
    throw std::runtime_error(
        "Number of point weights does not match the number of points.");
  }

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

  // Sample the local keys/weights, over-sampling by a fixed factor per
  // partition, and gather the samples on all ranks. Splitters taken at
  // equal cumulative-weight positions of the sorted sample then cut the
  // global key order into `nparts` pieces of approximately equal
  // weight, without sorting the keys across ranks.
  //
  // Note: the local points are sorted by key first so that the sample
  // is a systematic sample of the local (key, cumulative weight)
  // relationship. Sampling in incoming order instead makes the sample a
  // random one, whose much larger variance shows up directly as
  // partition imbalance (measured: 7% versus under 1% for a mesh whose
  // cells arrive in an order unrelated to their position).
  constexpr int oversample = 32;
  const std::size_t n
      = std::min(num_points, static_cast<std::size_t>(oversample) * nparts);

  std::vector<std::uint64_t> splitters(nparts - 1);
  if (!weights)
  {
    // Fast path: every point has equal weight, so the sample can be
    // taken directly from the sorted keys, by position, with no need
    // to track weights or an index permutation alongside them.
    std::vector<std::uint64_t> keys_sorted(keys);
    dolfinx::radix_sort(keys_sorted);
    std::vector<std::uint64_t> sample(n);
    for (std::size_t j = 0; j < n; ++j)
      sample[j] = keys_sorted[(j * num_points) / n];

    std::vector<std::uint64_t> sample_all;
    {
      // TODO: the merged sample count (displs.back()) can overflow int
      // for very large nparts/rank counts (~32 * nparts^2 samples in
      // the worst case), since MPI_Allgatherv's counts/displacements
      // are int. Would need either a sample budget that doesn't grow
      // with nparts^2 or the MPI-4 large-count (_c) collectives.
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

    for (int p = 1; p < nparts; ++p)
      splitters[p - 1] = sample_all[(sample_all.size() * p) / nparts];
  }
  else
  {
    std::vector<std::int32_t> order(num_points);
    std::iota(order.begin(), order.end(), 0);
    dolfinx::radix_sort(order, [&keys](std::int32_t i) { return keys[i]; });

    std::vector<std::int64_t> cum(num_points);
    std::int64_t local_total = 0;
    for (std::size_t i = 0; i < num_points; ++i)
    {
      local_total += (*weights)[order[i]];
      cum[i] = local_total;
    }

    std::vector<std::uint64_t> sample_keys(n);
    std::vector<double> sample_weights(n);
    for (std::size_t j = 0; j < n; ++j)
    {
      // Position of the sample in cumulative weight: the point whose
      // weight interval covers this position, i.e. the first point
      // whose inclusive cumulative weight exceeds it. With equal
      // weights this reduces to the point at array position `j *
      // num_points / n`. Computed in double, as for the global target
      // below, since local_total * j can exceed the range of
      // std::int64_t (large local point counts with weights near the
      // top of the std::int32_t range).
      const auto target = static_cast<std::int64_t>(
          static_cast<double>(local_total) * static_cast<double>(j) / n);
      auto it = std::ranges::upper_bound(cum, target);
      std::size_t pos = std::min(
          static_cast<std::size_t>(std::ranges::distance(cum.begin(), it)),
          num_points - 1);
      sample_keys[j] = keys[order[pos]];
      sample_weights[j] = static_cast<double>(local_total) / n;
    }

    std::vector<std::uint64_t> sample_keys_all;
    std::vector<double> sample_weights_all;
    {
      // TODO: see the same-shaped gather in the unweighted branch
      // above -- displs.back() can overflow int at large nparts/rank
      // counts.
      const int size = dolfinx::MPI::size(comm);
      int num_local = sample_keys.size();
      std::vector<int> counts(size), displs(size + 1, 0);
      MPI_Allgather(&num_local, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
      std::partial_sum(counts.begin(), counts.end(), std::next(displs.begin()));
      sample_keys_all.resize(displs.back());
      sample_weights_all.resize(displs.back());
      MPI_Allgatherv(sample_keys.data(), num_local, MPI_UINT64_T,
                     sample_keys_all.data(), counts.data(), displs.data(),
                     MPI_UINT64_T, comm);
      MPI_Allgatherv(sample_weights.data(), num_local, MPI_DOUBLE,
                     sample_weights_all.data(), counts.data(), displs.data(),
                     MPI_DOUBLE, comm);
    }

    // Degenerate case: no points anywhere
    if (sample_keys_all.empty())
      return std::vector<int>(num_points, 0);

    std::vector<std::int32_t> gorder(sample_keys_all.size());
    std::iota(gorder.begin(), gorder.end(), 0);
    dolfinx::radix_sort(gorder, [&sample_keys_all](std::int32_t i)
                        { return sample_keys_all[i]; });

    std::vector<double> gcum(gorder.size());
    double global_total = 0;
    for (std::size_t i = 0; i < gorder.size(); ++i)
    {
      global_total += sample_weights_all[gorder[i]];
      gcum[i] = global_total;
    }

    // Degenerate case: total weight of zero
    if (global_total <= 0)
      return std::vector<int>(num_points, 0);

    for (int p = 1; p < nparts; ++p)
    {
      const double target = global_total * static_cast<double>(p) / nparts;
      auto it = std::ranges::upper_bound(gcum, target);
      std::size_t pos = std::min(
          static_cast<std::size_t>(std::ranges::distance(gcum.begin(), it)),
          gorder.size() - 1);
      splitters[p - 1] = sample_keys_all[gorder[pos]];
    }
  }

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
std::vector<int> graph::partition_sfc_morton(
    MPI_Comm comm, int nparts, std::span<const double> x, int gdim,
    std::optional<std::span<const std::int32_t>> node_weights)
{
  common::Timer timer("Compute Morton SFC partition of points");
  return partition_by_curve(comm, nparts, x, gdim, morton_key, node_weights);
}
//-----------------------------------------------------------------------------
std::vector<int> graph::partition_sfc_hilbert(
    MPI_Comm comm, int nparts, std::span<const double> x, int gdim,
    std::optional<std::span<const std::int32_t>> node_weights)
{
  common::Timer timer("Compute Hilbert SFC partition of points");
  return partition_by_curve(comm, nparts, x, gdim, hilbert_key, node_weights);
}
//-----------------------------------------------------------------------------
