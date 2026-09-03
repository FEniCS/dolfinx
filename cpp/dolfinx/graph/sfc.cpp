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
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/sort.h>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
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

/// Rank count below which post-office routing uses PCX. PCX uses O(P) memory;
/// NBX is used above this limit to bound memory use.
constexpr int pcx_rank_limit = 10000;

/// Number of samples per output partition.
constexpr int oversample = 32;

/// @brief A plan for routing values to the MPI rank that "owns" them,
/// via a sparse neighbourhood exchange.
///
/// The neighbourhood communicator (`neigh_comm`) must be freed by the
/// caller with `MPI_Comm_free`.
struct RoutingPlan
{
  MPI_Comm neigh_comm;
  std::vector<std::int32_t> perm; // reorders local entries by owner rank
  std::vector<int> send_counts, send_disp; // sized to unique owners
  std::vector<int> recv_counts, recv_disp; // sized to discovered sources
};

/// @brief Compute a RoutingPlan that sends each local entry `i` to rank
/// `owner[i]`.
///
/// @note Collective.
RoutingPlan compute_routing_plan(MPI_Comm comm,
                                 std::span<const std::int32_t> owner)
{
  RoutingPlan plan;
  plan.perm.resize(owner.size());
  std::iota(plan.perm.begin(), plan.perm.end(), 0);
  dolfinx::radix_sort(plan.perm, [&owner](std::int32_t i) { return owner[i]; });

  std::vector<int> dest;
  {
    std::size_t i = 0;
    while (i < plan.perm.size())
    {
      std::int32_t r = owner[plan.perm[i]];
      dest.push_back(r);
      std::size_t j = i;
      while (j < plan.perm.size() and owner[plan.perm[j]] == r)
        ++j;
      plan.send_counts.push_back(static_cast<int>(j - i));
      i = j;
    }
  }

  const int size = dolfinx::MPI::size(comm);
  std::vector<int> src
      = size < pcx_rank_limit
            ? dolfinx::MPI::compute_graph_edges_pcx(comm, dest)
            : dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  std::ranges::sort(src);

  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &plan.neigh_comm);

  plan.recv_counts.resize(src.size());
  plan.send_counts.reserve(1); // ensure data is not a nullptr
  plan.recv_counts.reserve(1); // ensure data is not a nullptr
  MPI_Neighbor_alltoall(plan.send_counts.data(), 1, MPI_INT,
                        plan.recv_counts.data(), 1, MPI_INT, plan.neigh_comm);

  plan.send_disp.assign(dest.size() + 1, 0);
  std::partial_sum(plan.send_counts.begin(), plan.send_counts.end(),
                   std::next(plan.send_disp.begin()));
  plan.recv_disp.assign(src.size() + 1, 0);
  std::partial_sum(plan.recv_counts.begin(), plan.recv_counts.end(),
                   std::next(plan.recv_disp.begin()));

  return plan;
}

/// @brief Route (key, weight) pairs (one pair per RoutingPlan::perm
/// entry) to their owning ranks, in a single combined exchange since
/// both share the same routing plan. A key, reinterpreted as
/// std::int64_t, is exchanged alongside its weight in one
/// MPI_Neighbor_alltoallv call by doubling the plan's per-rank
/// counts/displacements, rather than issuing two separate calls.
std::vector<std::int64_t>
route_key_weight(const RoutingPlan& plan, std::span<const std::uint64_t> keys,
                 std::span<const std::int64_t> weights)
{
  std::vector<std::int64_t> send_buf(2 * plan.perm.size());
  for (std::size_t i = 0; i < plan.perm.size(); ++i)
  {
    send_buf[2 * i] = static_cast<std::int64_t>(keys[plan.perm[i]]);
    send_buf[2 * i + 1] = weights[plan.perm[i]];
  }

  auto scale2 = [](int c) { return 2 * c; };
  std::vector<int> send_counts(plan.send_counts.size());
  std::ranges::transform(plan.send_counts, send_counts.begin(), scale2);
  std::vector<int> send_disp(plan.send_disp.size());
  std::ranges::transform(plan.send_disp, send_disp.begin(), scale2);
  std::vector<int> recv_counts(plan.recv_counts.size());
  std::ranges::transform(plan.recv_counts, recv_counts.begin(), scale2);
  std::vector<int> recv_disp(plan.recv_disp.size());
  std::ranges::transform(plan.recv_disp, recv_disp.begin(), scale2);

  std::vector<std::int64_t> recv_buf(recv_disp.back());
  send_buf.reserve(1);    // ensure data is not a nullptr
  send_counts.reserve(1); // ensure data is not a nullptr
  recv_counts.reserve(1); // ensure data is not a nullptr
  recv_buf.reserve(1);    // ensure data is not a nullptr
  MPI_Neighbor_alltoallv(send_buf.data(), send_counts.data(), send_disp.data(),
                         MPI_INT64_T, recv_buf.data(), recv_counts.data(),
                         recv_disp.data(), MPI_INT64_T, plan.neigh_comm);

  return recv_buf;
}

/// @brief Assign each of `keys` an owning rank, by slicing
/// `[gmin, gmax]` into `size` equal-width, contiguous ranges (rank `r`
/// owns range `r`). Computed in `double`, so this is an approximate
/// (not exact-integer) partition of the key range -- acceptable here
/// since it only decides which rank does the work of locating a
/// splitter, not the splitter value itself.
std::vector<std::int32_t> key_range_owners(std::span<const std::uint64_t> keys,
                                           std::uint64_t gmin,
                                           std::uint64_t gmax, int size)
{
  const double width = static_cast<double>(gmax - gmin) + 1.0;
  std::vector<std::int32_t> owner(keys.size());
  for (std::size_t j = 0; j < keys.size(); ++j)
  {
    double frac = static_cast<double>(keys[j] - gmin) / width;
    owner[j] = std::clamp(static_cast<std::int32_t>(frac * size), 0, size - 1);
  }
  return owner;
}

/// @brief floor(total * p / n), without overflowing `std::int64_t`.
///
/// `total` must be non-negative, `n` must be positive, and `p` must be
/// in [0, n]. `total`, `p` and `n` are not otherwise bounded -- in
/// particular a call site summing point/weight counts across many
/// ranks can exceed `int` range -- so the intermediate product is
/// formed in 128 bits rather than relying on any tighter bound holding
/// at every call site.
std::int64_t scaled_target(std::int64_t total, std::int64_t p, std::int64_t n)
{
  using boost::multiprecision::uint128_t;
  return static_cast<std::int64_t>(uint128_t(total) * uint128_t(p)
                                   / uint128_t(n));
}

/// @brief Return the local share of a globally bounded sample.
///
/// Every rank with points receives one sample so that its contribution
/// to the weight distribution is retained. Remaining samples are
/// allocated proportionally to the number of additional local points.
std::size_t sample_size(MPI_Comm comm, std::size_t num_points, int nparts)
{
  const int rank = dolfinx::MPI::rank(comm);
  std::array<std::int64_t, 2> local_counts
      = {static_cast<std::int64_t>(num_points), num_points > 0 ? 1 : 0};
  std::array<std::int64_t, 2> global_counts;
  const std::int64_t local_extra_points
      = num_points > 0 ? static_cast<std::int64_t>(num_points) - 1 : 0;
  std::int64_t extra_offset = 0;

  // global_counts and extra_offset each depend only on this rank's local
  // input, not on each other, so start both collectives before waiting on
  // either -- one round-trip latency instead of two.
  std::array<MPI_Request, 2> requests;
  MPI_Iallreduce(local_counts.data(), global_counts.data(), 2, MPI_INT64_T,
                 MPI_SUM, comm, &requests[0]);
  MPI_Iexscan(&local_extra_points, &extra_offset, 1, MPI_INT64_T, MPI_SUM, comm,
              &requests[1]);
  MPI_Waitall(2, requests.data(), MPI_STATUSES_IGNORE);

  const std::int64_t total_points = global_counts[0];
  const std::int64_t num_nonempty_ranks = global_counts[1];
  const std::int64_t num_samples = std::min(
      total_points, std::max(static_cast<std::int64_t>(oversample) * nparts,
                             num_nonempty_ranks));
  const std::int64_t total_extra_points = total_points - num_nonempty_ranks;

  if (rank == 0)
    extra_offset = 0;

  if (num_points == 0)
    return 0;

  const std::int64_t extra_samples = num_samples - num_nonempty_ranks;
  return 1
         + (total_extra_points > 0
                ? scaled_target(extra_samples,
                                extra_offset + local_extra_points,
                                total_extra_points)
                      - scaled_target(extra_samples, extra_offset,
                                      total_extra_points)
                : 0);
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
/// @brief Validate point coordinates and return the number of points.
///
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point.
/// @return Number of points in `x`.
std::size_t check_point_coordinates(std::span<const double> x, int gdim)
{
  if (gdim < 1 or gdim > 3)
    throw std::runtime_error("Geometric dimension must be 1, 2 or 3.");
  if (x.size() % gdim != 0)
  {
    throw std::runtime_error(
        "Point coordinate array size is not a multiple of gdim.");
  }

  return x.size() / gdim;
}

/// @brief Compute point bounds as negated minima and maxima.
///
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point.
/// @param[in] num_points Number of points in `x`.
/// @return Bounds stored as negated minima followed by maxima.
std::array<double, 6> compute_point_extent(std::span<const double> x, int gdim,
                                           std::size_t num_points)
{
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

  return extent;
}

/// @brief Quantise points and compute their space-filling-curve keys.
///
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point.
/// @param[in] num_points Number of points in `x`.
/// @param[in] extent Point bounds as returned by compute_point_extent.
/// @param[in] key Function that computes a key from quantised coordinates.
/// @return A key for each point.
template <typename K>
std::vector<std::uint64_t>
compute_sfc_keys(std::span<const double> x, int gdim, std::size_t num_points,
                 const std::array<double, 6>& extent, K key)
{
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

  return keys;
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
  const std::size_t num_points = check_point_coordinates(x, gdim);
  if (nparts < 1)
    throw std::runtime_error("Number of partitions must be > 0.");
  if (weights and weights->size() != num_points)
  {
    throw std::runtime_error(
        "Number of point weights does not match the number of points.");
  }

  if (nparts == 1)
    return std::vector<int>(num_points, 0);

  std::array<double, 6> extent = compute_point_extent(x, gdim, num_points);
  std::array<double, 6> recv;
  MPI_Allreduce(extent.data(), recv.data(), 6, MPI_DOUBLE, MPI_MAX, comm);
  extent = recv;
  std::vector<std::uint64_t> keys
      = compute_sfc_keys(x, gdim, num_points, extent, key);

  // Sample the local keys/weights, using a global budget that is a
  // fixed factor per partition. A sample is kept for every non-empty
  // rank, with the remainder distributed in proportion to local point
  // counts. Splitters are then found at equal cumulative-weight
  // positions of the *distributed* sample, cutting the global key
  // order into `nparts` pieces of approximately equal weight, without
  // ever sorting the full key set across ranks.
  //
  // Note: the local points are sorted by key first so that the sample
  // is a systematic sample of the local (key, cumulative weight)
  // relationship. Sampling in incoming order instead makes the sample a
  // random one, whose much larger variance shows up directly as
  // partition imbalance (measured: 7% versus under 1% for a mesh whose
  // cells arrive in an order unrelated to their position).
  //
  // Splitters are located via a "post office" scheme rather than
  // gathering the sample to every rank (as an Allgatherv would): each
  // rank owns a slice of the key range and only the sample entries
  // that fall in it, so per-rank cost scales with the sample size
  // divided across ranks, not multiplied by the rank count.
  const int size = dolfinx::MPI::size(comm);
  const int rank = dolfinx::MPI::rank(comm);
  const std::size_t n = sample_size(comm, num_points, nparts);

  std::vector<std::uint64_t> sample_keys(n);
  // Exact per-sample representative weight (assigned below, after both
  // branches), summing to exactly local_total regardless of how evenly
  // n divides it. Used unconditionally -- local_total is num_points
  // when `weights` is absent -- so the cross-rank cumulative tracking
  // below is one exact-integer code path either way.
  std::vector<std::int64_t> sample_weights(n);
  std::int64_t local_total = 0;
  if (!weights)
  {
    // Every point has equal weight, so the sample can be taken
    // directly from the sorted keys, by position, with no need to
    // track weights or an index permutation alongside them.
    std::vector<std::uint64_t> keys_sorted(keys);
    dolfinx::radix_sort(keys_sorted);
    for (std::size_t j = 0; j < n; ++j)
      sample_keys[j] = keys_sorted[(j * num_points) / n];
    local_total = static_cast<std::int64_t>(num_points);
  }
  else
  {
    std::vector<std::int32_t> order(num_points);
    std::iota(order.begin(), order.end(), 0);
    dolfinx::radix_sort(order, [&keys](std::int32_t i) { return keys[i]; });

    bool nonnegative_weights = true;
    for (std::size_t i = 0; i < num_points; ++i)
    {
      const std::int32_t weight = (*weights)[order[i]];
      local_total += weight;
      nonnegative_weights = nonnegative_weights and weight >= 0;
    }

    if (nonnegative_weights)
    {
      // Targets are ordered, so scan the cumulative weights once rather
      // than materialising them and binary-searching for every sample.
      std::size_t pos = 0;
      std::int64_t cumulative = (*weights)[order[0]];
      for (std::size_t j = 0; j < n; ++j)
      {
        // First point whose cumulative weight exceeds this target
        // position (reduces to array position `j * num_points / n` for
        // equal weights). Computed in double, since local_total * j can
        // overflow std::int64_t -- harmless here, as it only picks which
        // local point becomes sample j, not a value needing exactness.
        const auto target = static_cast<std::int64_t>(
            static_cast<double>(local_total) * static_cast<double>(j) / n);
        while (pos + 1 < num_points and cumulative <= target)
          cumulative += (*weights)[order[++pos]];
        sample_keys[j] = keys[order[pos]];
      }
    }
    else
    {
      // Negative weights do not form a monotone cumulative distribution.
      // Retain the existing upper_bound behaviour for this unsupported
      // input rather than changing its observable result here.
      std::vector<std::int64_t> cum(num_points);
      std::partial_sum(order.begin(), order.end(), cum.begin(),
                       [&weights](std::int64_t sum, std::int32_t i)
                       { return sum + (*weights)[i]; });
      for (std::size_t j = 0; j < n; ++j)
      {
        const auto target = static_cast<std::int64_t>(
            static_cast<double>(local_total) * static_cast<double>(j) / n);
        auto it = std::ranges::upper_bound(cum, target);
        std::size_t pos = std::min(
            static_cast<std::size_t>(std::ranges::distance(cum.begin(), it)),
            num_points - 1);
        sample_keys[j] = keys[order[pos]];
      }
    }
  }
  // Sample j's weight is the target range it was actually selected to
  // cover -- [scaled_target(local_total, j, n), scaled_target(...,
  // j+1, n)) -- not just any split summing to local_total: the
  // position formula's rounding doesn't spread remainders evenly, so a
  // mismatched split shows up directly as partition imbalance.
  std::int64_t prev = 0;
  for (std::size_t j = 0; j < n; ++j)
  {
    std::int64_t next = scaled_target(local_total, static_cast<int>(j) + 1,
                                      static_cast<int>(n));
    sample_weights[j] = next - prev;
    prev = next;
  }

  // Reduce {-min, max} in one MPI_MAX, as for the bounding box above.
  // Keys fit in 63 bits, so reinterpreting as std::int64_t and negating
  // is always safe. A rank with no local sample contributes sentinels
  // that always lose the reduction (INT64_MAX negated for the min
  // side, INT64_MIN, never negated, for the max side), so if every
  // rank is empty the reduced max stays negative -- impossible for a
  // real key -- flagging the degenerate case before casting back to
  // std::uint64_t.
  std::array<std::int64_t, 2> local_bounds
      = {n > 0 ? -static_cast<std::int64_t>(sample_keys.front())
               : -std::numeric_limits<std::int64_t>::max(),
         n > 0 ? static_cast<std::int64_t>(sample_keys.back())
               : std::numeric_limits<std::int64_t>::min()};
  std::array<std::int64_t, 2> global_bounds;
  MPI_Allreduce(local_bounds.data(), global_bounds.data(), 2, MPI_INT64_T,
                MPI_MAX, comm);

  // Degenerate case: no points anywhere
  if (global_bounds[1] < 0)
    return std::vector<int>(num_points, 0);

  std::uint64_t gmin = static_cast<std::uint64_t>(-global_bounds[0]);
  std::uint64_t gmax = static_cast<std::uint64_t>(global_bounds[1]);

  std::vector<std::int32_t> owner
      = key_range_owners(sample_keys, gmin, gmax, size);
  RoutingPlan plan = compute_routing_plan(comm, owner);
  std::vector<std::int64_t> recv_buf = route_key_weight(
      plan, sample_keys, std::span<const std::int64_t>(sample_weights));
  MPI_Comm_free(&plan.neigh_comm);

  // route_key_weight() groups by destination rank, not by key, so the receiver
  // must sort by key itself before it can walk cumulative weight.
  std::vector<std::int32_t> rorder(recv_buf.size() / 2);
  std::iota(rorder.begin(), rorder.end(), 0);
  dolfinx::radix_sort(rorder,
                      [&recv_buf](std::int32_t i) { return recv_buf[2 * i]; });

  std::int64_t local_weight = 0;
  for (std::size_t i = 0; i < rorder.size(); ++i)
    local_weight += recv_buf[2 * rorder[i] + 1];

  // Rank order matches key-range order (see key_range_owners), so an
  // exclusive scan of local_weight gives each rank its offset into the
  // virtual, fully key-sorted, fully merged sample. Exact integer
  // arithmetic matters here: a scan over floating-point weights isn't
  // guaranteed bit-for-bit consistent across ranks, and could leave a
  // target unclaimed, or claimed twice, at a rank boundary.
  // offset and total_weight both reduce the same local_weight value but
  // don't depend on each other, so overlap them as above.
  std::int64_t offset = 0;
  std::int64_t total_weight = 0;
  std::array<MPI_Request, 2> weight_requests;
  MPI_Iexscan(&local_weight, &offset, 1, MPI_INT64_T, MPI_SUM, comm,
              &weight_requests[0]);
  MPI_Iallreduce(&local_weight, &total_weight, 1, MPI_INT64_T, MPI_SUM, comm,
                 &weight_requests[1]);
  MPI_Waitall(2, weight_requests.data(), MPI_STATUSES_IGNORE);
  if (rank == 0)
    offset = 0; // Exscan leaves rank 0's recvbuf value unspecified

  // Degenerate case: total weight of zero
  if (total_weight <= 0)
    return std::vector<int>(num_points, 0);

  // Each target lands in exactly one rank's [offset, offset+local_weight)
  // range; contributions are otherwise left at 0 so the final Allreduce
  // can just sum them.
  std::vector<std::uint64_t> contrib(nparts - 1, 0);
  const bool nonnegative_sample_weights = std::ranges::all_of(
      rorder, [&recv_buf](std::int32_t i) { return recv_buf[2 * i + 1] >= 0; });
  std::vector<std::int64_t> rcum;
  if (!nonnegative_sample_weights)
  {
    rcum.resize(rorder.size());
    std::partial_sum(rorder.begin(), rorder.end(), rcum.begin(),
                     [&recv_buf](std::int64_t sum, std::int32_t i)
                     { return sum + recv_buf[2 * i + 1]; });
  }

  std::size_t pos = 0;
  std::int64_t cumulative = rorder.empty() ? 0 : recv_buf[2 * rorder[0] + 1];
  for (int p = 1; p < nparts; ++p)
  {
    std::int64_t target = scaled_target(total_weight, p, nparts);
    if (target >= offset and target < offset + local_weight)
    {
      if (nonnegative_sample_weights)
      {
        while (pos + 1 < rorder.size() and cumulative <= target - offset)
          cumulative += recv_buf[2 * rorder[++pos] + 1];
        contrib[p - 1] = static_cast<std::uint64_t>(recv_buf[2 * rorder[pos]]);
      }
      else
      {
        auto it = std::ranges::upper_bound(rcum, target - offset);
        std::size_t i = std::min(
            static_cast<std::size_t>(std::ranges::distance(rcum.begin(), it)),
            rorder.size() - 1);
        contrib[p - 1] = static_cast<std::uint64_t>(recv_buf[2 * rorder[i]]);
      }
    }
  }
  std::vector<std::uint64_t> splitters(nparts - 1);
  MPI_Allreduce(contrib.data(), splitters.data(), nparts - 1, MPI_UINT64_T,
                MPI_SUM, comm);

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
/// @brief Reorder points by their space-filling-curve keys.
///
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point.
/// @param[in] key Function that computes a key from quantised coordinates.
/// @return Reordering map from each input point to its output index.
template <typename K>
std::vector<std::int32_t> reorder_by_curve(std::span<const double> x, int gdim,
                                           K key)
{
  const std::size_t num_points = check_point_coordinates(x, gdim);
  const std::array<double, 6> extent
      = compute_point_extent(x, gdim, num_points);
  std::vector<std::uint64_t> keys
      = compute_sfc_keys(x, gdim, num_points, extent, key);

  std::vector<std::int32_t> order(num_points);
  std::iota(order.begin(), order.end(), 0);
  dolfinx::radix_sort(order, [&keys](std::int32_t i) { return keys[i]; });

  std::vector<std::int32_t> map(num_points);
  for (std::size_t i = 0; i < order.size(); ++i)
    map[order[i]] = i;
  return map;
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
std::vector<std::int32_t> graph::reorder_sfc_morton(std::span<const double> x,
                                                    int gdim)
{
  common::Timer timer("Compute Morton SFC ordering of points");
  return reorder_by_curve(x, gdim, morton_key);
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> graph::reorder_sfc_hilbert(std::span<const double> x,
                                                     int gdim)
{
  common::Timer timer("Compute Hilbert SFC ordering of points");
  return reorder_by_curve(x, gdim, hilbert_key);
}
//-----------------------------------------------------------------------------
