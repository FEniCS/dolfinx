// Copyright (C) 2020-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "partition.h"
#include "AdjacencyList.h"
#include "partitioners.h"
#include <algorithm>
#include <array>
#include <boost/sort/sort.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <functional>
#include <limits>
#include <memory>
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
/// @brief Partition points by their position along a space-filling curve
/// and, if requested, add the destinations needed for ghosting.
///
/// @param[in] comm MPI communicator the points are distributed across.
/// @param[in] nparts Number of partitions.
/// @param[in] graph Node connectivity graph, one node per point. Used
/// only to determine ghost nodes.
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point.
/// @param[in] ghosting Flag to enable ghosting.
/// @param[in] key Curve key for a point.
/// @return Destination rank(s) for each point, owner first.
template <typename K>
graph::AdjacencyList<std::int32_t>
partition_curve(MPI_Comm comm, int nparts,
                const graph::AdjacencyList<std::int64_t>& graph,
                std::span<const double> x, int gdim, bool ghosting, K key)
{
  if (static_cast<std::int64_t>(x.size())
      != static_cast<std::int64_t>(gdim) * graph.num_nodes())
  {
    throw std::runtime_error(
        "Number of coordinates does not match number of graph nodes.");
  }

  std::vector<int> part = partition_by_curve(comm, nparts, x, gdim, key);
  if (!ghosting)
    return dolfinx::graph::regular_adjacency_list(std::move(part), 1);

  // Wherever a point goes, so must the points connected to it by an edge
  std::vector<int> node_disp(dolfinx::MPI::size(comm) + 1, 0);
  const int num_local = graph.num_nodes();
  MPI_Allgather(&num_local, 1, MPI_INT, std::next(node_disp.data()), 1, MPI_INT,
                comm);
  std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
  return dolfinx::graph::compute_destination_ranks(comm, graph, node_disp,
                                                   part);
}
} // namespace

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
graph::partition_sfc_morton(MPI_Comm comm, int nparts,
                            const graph::AdjacencyList<std::int64_t>& graph,
                            std::span<const double> x, int gdim, bool ghosting)
{
  common::Timer timer("Compute Morton SFC partition of points");
  return partition_curve(comm, nparts, graph, x, gdim, ghosting, morton_key);
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
graph::partition_sfc_hilbert(MPI_Comm comm, int nparts,
                             const graph::AdjacencyList<std::int64_t>& graph,
                             std::span<const double> x, int gdim, bool ghosting)
{
  common::Timer timer("Compute Hilbert SFC partition of points");
  return partition_curve(comm, nparts, graph, x, gdim, ghosting, hilbert_key);
}
//-----------------------------------------------------------------------------
graph::partition_fn graph::sfc::partitioner(sfc::curve curve)
{
  return
      [curve](
          MPI_Comm comm, int nparts,
          std::optional<
              std::reference_wrapper<const graph::AdjacencyList<std::int64_t>>>
              local_graph,
          std::optional<std::span<const double>> x, int gdim, bool ghosting)
  {
    if (!x)
    {
      throw std::runtime_error(
          "Space-filling curve partitioner requires point coordinates.");
    }
    if (ghosting and !local_graph)
    {
      throw std::runtime_error("Space-filling curve partitioner requires a "
                               "graph to compute ghosts.");
    }

    auto call = [&](const graph::AdjacencyList<std::int64_t>& graph)
    {
      return (curve == sfc::curve::hilbert)
                 ? graph::partition_sfc_hilbert(comm, nparts, graph, *x, gdim,
                                                ghosting)
                 : graph::partition_sfc_morton(comm, nparts, graph, *x, gdim,
                                               ghosting);
    };

    if (local_graph)
      return call(local_graph->get());

    // local_graph is not read at all when ghosting is false, so a
    // trivial placeholder of the right size stands in for it.
    const graph::AdjacencyList<std::int64_t> trivial_graph(
        static_cast<std::int32_t>(x->size() / gdim));
    return call(trivial_graph);
  };
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
graph::partition_graph(MPI_Comm comm, int nparts,
                       const AdjacencyList<std::int64_t>& local_graph,
                       bool ghosting)
{
#if HAS_PARMETIS
  return graph::parmetis::partitioner()(comm, nparts, std::cref(local_graph),
                                        std::nullopt, 0, ghosting);
#elif HAS_PTSCOTCH
  return graph::scotch::partitioner()(comm, nparts, std::cref(local_graph),
                                      std::nullopt, 0, ghosting);
#elif HAS_KAHIP
  return graph::kahip::partitioner()(comm, nparts, std::cref(local_graph),
                                     std::nullopt, 0, ghosting);
#else
// Should never reach this point
#endif
}
//-----------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
graph::build::distribute(MPI_Comm comm,
                         const graph::AdjacencyList<std::int64_t>& list,
                         const graph::AdjacencyList<std::int32_t>& destinations)
{
  common::Timer timer("Distribute AdjacencyList nodes to destination ranks");

  assert(list.num_nodes() == (int)destinations.num_nodes());
  const int rank = dolfinx::MPI::rank(comm);

  // Get global offset for converting local index to global index for
  // nodes in 'list'
  std::int64_t offset_global = 0;
  {
    const std::int64_t num_owned = list.num_nodes();
    MPI_Exscan(&num_owned, &offset_global, 1, MPI_INT64_T, MPI_SUM, comm);
  }

  // TODO: Do this on the neighbourhood only
  // Get the maximum number of edges for a node
  int shape1 = 0;
  {
    int shape1_local = list.num_nodes() > 0 ? list.links(0).size() : 0;
    MPI_Allreduce(&shape1_local, &shape1, 1, MPI_INT, MPI_MAX, comm);
  }

  // Buffer size (max number of edges + 3 for num_edges, owning rank,
  // and node global index)
  const std::size_t buffer_shape1 = shape1 + 3;

  // Build (dest, index, owning rank) list and sort
  std::vector<std::array<int, 3>> dest_to_index;
  dest_to_index.reserve(destinations.array().size());
  for (std::int32_t i = 0; i < destinations.num_nodes(); ++i)
  {
    auto di = destinations.links(i);
    std::ranges::transform(di, std::back_inserter(dest_to_index),
                           [i, d0 = di.front()](auto d) -> std::array<int, 3>
                           { return {d, i, d0}; });
  }

  // Only grouping by destination rank is required (order within a group
  // is irrelevant downstream), and the key is bounded by the
  // communicator size, so a radix sort keyed on the destination rank
  // alone is used rather than a full lexicographic sort.
  dolfinx::radix_sort(dest_to_index, [](const auto& e) { return e[0]; });

  // Build list of unique dest ranks and count number of rows to send to
  // each dest (by neighbourhood rank)
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest;
  {
    auto it = dest_to_index.begin();
    while (it != dest_to_index.end())
    {
      // Store global rank and find iterator to next global rank
      dest.push_back(it->front());
      auto it1
          = std::find_if(it, dest_to_index.end(),
                         [r = dest.back()](auto idx) { return idx[0] != r; });

      // Store number of items for current rank
      num_items_per_dest.push_back(std::ranges::distance(it, it1));

      // Advance iterator
      it = it1;
    }
  }

  // Determine source ranks. Sort ranks to make distribution
  // deterministic.
  std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  std::ranges::sort(src);

  // Create neighbourhood communicator
  MPI_Comm neigh_comm;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm);

  // Send number of nodes to receivers
  std::vector<int> num_items_recv(src.size());
  num_items_per_dest.reserve(1);
  num_items_recv.reserve(1);
  MPI_Request request_size;
  MPI_Ineighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                         num_items_recv.data(), 1, MPI_INT, neigh_comm,
                         &request_size);

  // Compute send displacements
  std::vector<std::int32_t> send_disp(num_items_per_dest.size() + 1, 0);
  std::partial_sum(num_items_per_dest.begin(), num_items_per_dest.end(),
                   std::next(send_disp.begin()));

  // Pack send buffer
  std::vector<std::int64_t> send_buffer(buffer_shape1 * send_disp.back(), -1);
  {
    assert(send_disp.back() == (std::int32_t)dest_to_index.size());
    for (std::size_t i = 0; i < dest_to_index.size(); ++i)
    {
      std::array<int, 3> dest_data = dest_to_index[i];
      const std::size_t pos = dest_data[1];

      std::span b(send_buffer.data() + i * buffer_shape1, buffer_shape1);
      auto row = list.links(pos);
      std::ranges::copy(row, b.begin());

      auto info = b.last(3);
      info[0] = row.size();          // Number of edges for node
      info[1] = dest_data[2];        // Owning rank
      info[2] = pos + offset_global; // Original global index
    }
  }

  // Prepare receive displacement
  MPI_Wait(&request_size, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_disp(num_items_recv.size() + 1, 0);
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::next(recv_disp.begin()));

  // Send/receive data facet
  MPI_Datatype compound_type;
  MPI_Type_contiguous(buffer_shape1, MPI_INT64_T, &compound_type);
  MPI_Type_commit(&compound_type);
  std::vector<std::int64_t> recv_buffer(buffer_shape1 * recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer.data(), num_items_per_dest.data(),
                         send_disp.data(), compound_type, recv_buffer.data(),
                         num_items_recv.data(), recv_disp.data(), compound_type,
                         neigh_comm);
  MPI_Type_free(&compound_type);
  MPI_Comm_free(&neigh_comm);

  // Unpack receive buffer
  std::vector<int> src_ranks, src_ranks1, ghost_index_owner;
  src_ranks.reserve(recv_disp.back());
  src_ranks1.reserve(recv_disp.back());

  std::vector<std::int64_t> data, data1;
  data.reserve((buffer_shape1 - 3) * recv_disp.back());
  data1.reserve((buffer_shape1 - 3) * recv_disp.back());

  std::vector<std::int32_t> offsets{0}, offsets1{0};
  offsets.reserve(recv_disp.back());
  offsets1.reserve(recv_disp.back());

  std::vector<std::int64_t> global_indices, global_indices1;
  global_indices.reserve(recv_disp.back());
  global_indices1.reserve(recv_disp.back());
  for (std::size_t p = 0; p < recv_disp.size() - 1; ++p)
  {
    const int src_rank = src[p];
    for (std::int32_t i = recv_disp[p]; i < recv_disp[p + 1]; ++i)
    {
      std::span row(recv_buffer.data() + i * buffer_shape1, buffer_shape1);
      auto info = row.last(3);
      std::size_t num_edges = info[0];
      std::int64_t orig_global_index = info[2];
      auto edges = row.first(num_edges);
      if (int owner = info[1]; owner == rank)
      {
        data.insert(data.end(), edges.begin(), edges.end());
        offsets.push_back(offsets.back() + num_edges);
        src_ranks.push_back(src_rank);
        global_indices.push_back(orig_global_index);
      }
      else
      {
        data1.insert(data1.end(), edges.begin(), edges.end());
        offsets1.push_back(offsets1.back() + info[0]);
        src_ranks1.push_back(src_rank);
        global_indices1.push_back(orig_global_index);
        ghost_index_owner.push_back(info[1]);
      }
    }
  }

  std::ranges::transform(offsets1, offsets1.begin(),
                         [off = offsets.back()](auto x) { return x + off; });
  data.insert(data.end(), data1.begin(), data1.end());
  offsets.insert(offsets.end(), std::next(offsets1.begin()), offsets1.end());
  src_ranks.insert(src_ranks.end(), src_ranks1.begin(), src_ranks1.end());
  global_indices.insert(global_indices.end(), global_indices1.begin(),
                        global_indices1.end());

  data.shrink_to_fit();
  offsets.shrink_to_fit();
  src_ranks.shrink_to_fit();
  global_indices.shrink_to_fit();
  ghost_index_owner.shrink_to_fit();

  return {
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offsets)),
      std::move(src_ranks), std::move(global_indices),
      std::move(ghost_index_owner)};
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<std::int64_t>, std::vector<int>,
           std::vector<std::int64_t>, std::vector<int>>
graph::build::distribute(MPI_Comm comm, std::span<const std::int64_t> list,
                         std::array<std::size_t, 2> shape,
                         const graph::AdjacencyList<std::int32_t>& destinations)
{
  common::Timer timer(
      "Distribute fixed-degree adjacency list to destination ranks");

  assert(list.size() == shape[0] * shape[1]);
  assert(destinations.num_nodes() == (std::int32_t)shape[0]);
  int rank = dolfinx::MPI::rank(comm);
  std::int64_t num_owned = destinations.num_nodes();

  // Get global offset for converting local index to global index for
  // nodes in 'list'
  std::int64_t offset_global = 0;
  MPI_Exscan(&num_owned, &offset_global, 1, MPI_INT64_T, MPI_SUM, comm);

  // Buffer size (max number of edges + 2 for owning rank,
  // and node global index)
  const std::size_t buffer_shape1 = shape[1] + 2;

  // Build (dest, index, owning rank) list and sort
  std::vector<std::array<int, 3>> dest_to_index;
  dest_to_index.reserve(destinations.array().size());
  for (std::int32_t i = 0; i < destinations.num_nodes(); ++i)
  {
    auto di = destinations.links(i);
    std::ranges::transform(di, std::back_inserter(dest_to_index),
                           [i, d0 = di.front()](auto d) -> std::array<int, 3>
                           { return {d, i, d0}; });
  }

  // Only grouping by destination rank is required (order within a group
  // is irrelevant downstream), and the key is bounded by the
  // communicator size, so a radix sort keyed on the destination rank
  // alone is used rather than a full lexicographic sort.
  dolfinx::radix_sort(dest_to_index, [](const auto& e) { return e[0]; });

  // Build list of unique dest ranks and count number of rows to send to
  // each dest (by neighbourhood rank)
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest;
  {
    auto it = dest_to_index.begin();
    while (it != dest_to_index.end())
    {
      // Store global rank and find iterator to next global rank
      dest.push_back(it->front());
      auto it1
          = std::find_if(it, dest_to_index.end(),
                         [r = dest.back()](auto& idx) { return idx[0] != r; });

      // Store number of items for current rank
      num_items_per_dest.push_back(std::ranges::distance(it, it1));

      // Advance iterator
      it = it1;
    }
  }

  // Determine source ranks. Sort ranks to make distribution
  // deterministic.
  std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  std::ranges::sort(src);

  // Create neighbourhood communicator
  MPI_Comm neigh_comm;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm);

  // Send number of nodes to receivers
  std::vector<int> num_items_recv(src.size());
  num_items_per_dest.reserve(1);
  num_items_recv.reserve(1);
  MPI_Request request_size;
  MPI_Ineighbor_alltoall(num_items_per_dest.data(), 1, MPI_INT,
                         num_items_recv.data(), 1, MPI_INT, neigh_comm,
                         &request_size);

  // Compute send displacements
  std::vector<std::int32_t> send_disp(num_items_per_dest.size() + 1, 0);
  std::partial_sum(num_items_per_dest.begin(), num_items_per_dest.end(),
                   std::next(send_disp.begin()));

  // Pack send buffer
  std::vector<std::int64_t> send_buffer(buffer_shape1 * send_disp.back(), -1);
  {
    assert(send_disp.back() == (std::int32_t)dest_to_index.size());
    for (std::size_t i = 0; i < dest_to_index.size(); ++i)
    {
      std::array<int, 3> dest_data = dest_to_index[i];
      const std::size_t pos = dest_data[1];

      std::span b(send_buffer.data() + i * buffer_shape1, buffer_shape1);
      std::span row(list.data() + pos * shape[1], shape[1]);
      std::ranges::copy(row, b.begin());

      auto info = b.last(2);
      info[0] = dest_data[2];        // Owning rank
      info[1] = pos + offset_global; // Original global index
    }
  }

  // Prepare receive displacement
  MPI_Wait(&request_size, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_disp(num_items_recv.size() + 1, 0);
  std::partial_sum(num_items_recv.begin(), num_items_recv.end(),
                   std::next(recv_disp.begin()));

  // Send/receive data facet
  MPI_Datatype compound_type;
  MPI_Type_contiguous(buffer_shape1, MPI_INT64_T, &compound_type);
  MPI_Type_commit(&compound_type);
  std::vector<std::int64_t> recv_buffer(buffer_shape1 * recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer.data(), num_items_per_dest.data(),
                         send_disp.data(), compound_type, recv_buffer.data(),
                         num_items_recv.data(), recv_disp.data(), compound_type,
                         neigh_comm);
  MPI_Type_free(&compound_type);
  MPI_Comm_free(&neigh_comm);

  spdlog::debug("Received {} data on {} [{}]", recv_disp.back(), rank,
                shape[1]);

  // Unpack receive buffer
  std::vector<std::int64_t> data, data1;
  std::vector<int> ghost_index_owner;
  std::vector<std::int64_t> global_indices, global_indices1;
  std::vector<int> src_ranks, src_ranks1;
  for (std::size_t p = 0; p < recv_disp.size() - 1; ++p)
  {
    int src_rank = src[p];
    for (std::int32_t q = recv_disp[p]; q < recv_disp[p + 1]; ++q)
    {
      std::span row(recv_buffer.data() + q * buffer_shape1, buffer_shape1);
      auto info = row.last(2);
      std::int64_t orig_global_index = info[1];
      auto edges = row.first(shape[1]);
      if (int owner = info[0]; owner == rank)
      {
        data.insert(data.end(), edges.begin(), edges.end());
        global_indices.push_back(orig_global_index);
        src_ranks.push_back(src_rank);
      }
      else
      {
        data1.insert(data1.end(), edges.begin(), edges.end());
        global_indices1.push_back(orig_global_index);
        ghost_index_owner.push_back(owner);
        src_ranks1.push_back(src_rank);
      }
    }
  }

  data.insert(data.end(), data1.begin(), data1.end());
  data.shrink_to_fit();
  global_indices.insert(global_indices.end(), global_indices1.begin(),
                        global_indices1.end());
  global_indices.shrink_to_fit();
  src_ranks.insert(src_ranks.end(), src_ranks1.begin(), src_ranks1.end());
  src_ranks.shrink_to_fit();
  ghost_index_owner.shrink_to_fit();

  return {std::move(data), std::move(src_ranks), std::move(global_indices),
          std::move(ghost_index_owner)};
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> graph::build::compute_ghost_indices(
    MPI_Comm comm, std::span<const std::int64_t> owned_indices,
    std::span<const std::int64_t> ghost_indices,
    std::span<const int> ghost_owners, int num_threads)
{
  common::Timer timer("Compute ghost indices");
  spdlog::info("Compute ghost indices");

  // Get number of local cells determine global offset
  std::int64_t offset_local = 0;
  MPI_Request request_offset_scan;
  const std::int64_t num_local = owned_indices.size();
  MPI_Iexscan(&num_local, &offset_local, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_offset_scan);

  // Find out how many ghosts are on each neighboring process
  std::vector<int> ghost_index_count;
  std::vector<int> neighbors;
  // A tree map here costs a heap-allocating node lookup/insert on
  // every one of potentially many millions of ghost ranks touched
  // below, even though the map itself stays tiny (bounded by the
  // number of distinct neighbour ranks); an open-addressed map avoids
  // that per-element allocation and pointer-chasing.
  boost::unordered_flat_map<int, int> proc_to_neighbor;
  for (int p : ghost_owners)
  {
    assert(p != dolfinx::MPI::rank(comm));
    auto [it, insert] = proc_to_neighbor.insert({p, neighbors.size()});
    if (insert)
    {
      // New neighbor found
      neighbors.push_back(p);
      ghost_index_count.push_back(0);
    }
    ++ghost_index_count[it->second];
  }

  MPI_Comm neighbor_comm_fwd, neighbor_comm_rev;

  std::vector<int> in_edges
      = dolfinx::MPI::compute_graph_edges_pcx(comm, neighbors);
  MPI_Dist_graph_create_adjacent(comm, in_edges.size(), in_edges.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbor_comm_fwd);
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, in_edges.size(),
                                 in_edges.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
                                 false, &neighbor_comm_rev);

  std::vector<int> send_offsets{0};
  send_offsets.reserve(ghost_index_count.size() + 1);
  std::partial_sum(ghost_index_count.begin(), ghost_index_count.end(),
                   std::back_inserter(send_offsets));

  // Copy offsets to help fill array
  std::vector<std::int64_t> send_data(send_offsets.back());
  {
    std::vector<int> ghost_index_offset = send_offsets;
    for (std::size_t i = 0; i < ghost_owners.size(); ++i)
    {
      // Owning process
      int p = ghost_owners[i];

      // Owning neighbor
      int np = proc_to_neighbor[p];

      // Send data location
      int pos = ghost_index_offset[np];
      send_data[pos] = ghost_indices[i];
      ++ghost_index_offset[np];
    }
  }

  std::vector<int> recv_sizes(in_edges.size());
  ghost_index_count.reserve(1);
  recv_sizes.reserve(1);

  MPI_Neighbor_alltoall(ghost_index_count.data(), 1, MPI_INT, recv_sizes.data(),
                        1, MPI_INT, neighbor_comm_fwd);

  std::vector<int> recv_offsets{0};
  recv_offsets.reserve(recv_sizes.size() + 1);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::back_inserter(recv_offsets));

  std::vector<std::int64_t> recv_data(recv_offsets.back());
  MPI_Neighbor_alltoallv(send_data.data(), ghost_index_count.data(),
                         send_offsets.data(), MPI_INT64_T, recv_data.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT64_T,
                         neighbor_comm_fwd);

  // Complete global_offset scan
  MPI_Wait(&request_offset_scan, MPI_STATUS_IGNORE);

  if (num_threads > 1)
  {
    std::vector<std::array<std::int64_t, 2>> old_to_new;
    old_to_new.reserve(owned_indices.size());
    for (auto idx : owned_indices)
    {
      old_to_new.push_back(
          {idx, static_cast<std::int64_t>(offset_local + old_to_new.size())});
    }

    boost::sort::block_indirect_sort(old_to_new.begin(), old_to_new.end(),
                                     num_threads);

    // Replace values in recv_data with new_index and send back
    std::ranges::transform(
        recv_data, recv_data.begin(),
        [&old_to_new](auto r)
        {
          auto it = std::ranges::lower_bound(old_to_new, r, std::ranges::less(),
                                             [](auto e) { return e[0]; });
          assert(it != old_to_new.end() and it->front() == r);
          return (*it)[1];
        });
  }
  else
  {
    // Map from old (global) index to new (global) index. A hash map
    // replaces the sort and O(log n) binary search with an
    // O(1)-average lookup per entry of recv_data -- worth it since
    // owned_indices (built once) can be in the millions, while
    // recv_data (the ghost-boundary traffic, looked up repeatedly) is
    // typically far smaller.
    boost::unordered_flat_map<std::int64_t, std::int64_t> old_to_new;
    old_to_new.reserve(owned_indices.size());
    std::int64_t new_idx = offset_local;
    for (auto idx : owned_indices)
      old_to_new.emplace(idx, new_idx++);

    // Replace values in recv_data with new_index and send back
    std::ranges::transform(recv_data, recv_data.begin(),
                           [&old_to_new](auto r)
                           {
                             auto it = old_to_new.find(r);
                             assert(it != old_to_new.end());
                             return it->second;
                           });
  }

  std::vector<std::int64_t> new_recv(send_data.size());
  MPI_Neighbor_alltoallv(recv_data.data(), recv_sizes.data(),
                         recv_offsets.data(), MPI_INT64_T, new_recv.data(),
                         ghost_index_count.data(), send_offsets.data(),
                         MPI_INT64_T, neighbor_comm_rev);
  MPI_Comm_free(&neighbor_comm_fwd);
  MPI_Comm_free(&neighbor_comm_rev);

  // Build old id -> new id map. As above, built once and then queried
  // once per entry of `ghost_indices` -- a hash map replaces the sort
  // and the O(log n) binary search per lookup with an O(1)-average
  // lookup.
  boost::unordered_flat_map<std::int64_t, std::int64_t> old_to_new1;
  old_to_new1.reserve(send_data.size());
  for (std::size_t i = 0; i < send_data.size(); ++i)
    old_to_new1.emplace(send_data[i], new_recv[i]);

  std::vector<std::int64_t> ghost_global_indices(ghost_indices.size());
  std::ranges::transform(ghost_indices, ghost_global_indices.begin(),
                         [&old_to_new1](auto q)
                         {
                           auto it = old_to_new1.find(q);
                           assert(it != old_to_new1.end());
                           return it->second;
                         });

  return ghost_global_indices;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
graph::build::compute_local_to_global(std::span<const std::int64_t> global,
                                      std::span<const std::int32_t> local)
{
  common::Timer timer(
      "Compute-local-to-global links for global/local adjacency list");

  if (global.empty() and local.empty())
    return std::vector<std::int64_t>();
  if (global.size() != local.size())
    throw std::runtime_error("Data size mismatch.");

  std::int32_t max_local_idx = *std::ranges::max_element(local);
  std::vector<std::int64_t> local_to_global_list(max_local_idx + 1, -1);
  for (std::size_t i = 0; i < local.size(); ++i)
  {
    if (local_to_global_list[local[i]] == -1)
      local_to_global_list[local[i]] = global[i];
  }

  return local_to_global_list;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> graph::build::compute_local_to_local(
    std::span<const std::int64_t> local0_to_global,
    std::span<const std::int64_t> local1_to_global)
{
  common::Timer timer("Compute local-to-local map");
  assert(local0_to_global.size() == local1_to_global.size());

  // Compute inverse map for local1_to_global
  std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local1;
  global_to_local1.reserve(local1_to_global.size());
  for (auto idx_global : local1_to_global)
    global_to_local1.push_back({idx_global, global_to_local1.size()});
  std::ranges::sort(global_to_local1);

  // If global_to_local1 is contiguous starting at 0 (its .first values
  // are sorted and unique, so this is easy to detect and implies
  // position == .first), reading .second directly avoids a
  // binary-search lookup for every element of local0_to_global below.
  // Every value looked up is guaranteed present in global_to_local1 by
  // this function's precondition, so - given is_identity - always
  // within bounds; the bounds check is a defensive no-op fallback
  // rather than something expected to trigger.
  const bool is_identity
      = !global_to_local1.empty() and global_to_local1.front().first == 0
        and global_to_local1.back().first
                == static_cast<std::int64_t>(global_to_local1.size()) - 1;

  // Compute inverse map for local0_to_local1
  std::vector<std::int32_t> local0_to_local1;
  local0_to_local1.reserve(local0_to_global.size());
  std::ranges::transform(
      local0_to_global, std::back_inserter(local0_to_local1),
      [&global_to_local1, is_identity](auto l2g)
      {
        if (is_identity and std::size_t(l2g) < global_to_local1.size())
          return global_to_local1[l2g].second;

        auto it = std::ranges::lower_bound(global_to_local1, l2g,
                                           std::ranges::less(),
                                           [](auto e) { return e.first; });
        assert(it != global_to_local1.end() and it->first == l2g);
        return it->second;
      });

  return local0_to_local1;
}
//-----------------------------------------------------------------------------
