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
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace dolfinx;

namespace
{
/// @brief Setup shared by both graph::build::distribute overloads:
/// determine send/receive ranks and start exchanging item counts,
/// before each overload packs and exchanges its own payload.
///
/// The size exchange (`request`) is left pending so the caller can
/// overlap it with (potentially expensive) send-buffer packing.
struct DistributionPlan
{
  /// (destination rank, local index, owning rank) triples for each
  /// entry of `destinations`, one per outgoing edge, sorted by
  /// destination rank.
  std::vector<std::array<int, 3>> dest_to_index;
  /// Unique destination ranks, i.e. the neighbourhood communicator's
  /// out-edges.
  std::vector<int> dest;
  /// Number of rows sent to each entry of `dest`.
  std::vector<std::int32_t> num_items_per_dest;
  /// Send displacements into `dest_to_index`, size `dest.size() + 1`.
  std::vector<std::int32_t> send_disp;
  /// Source ranks, i.e. the neighbourhood communicator's in-edges.
  std::vector<int> src;
  /// Number of rows received from each entry of `src`. Filled once
  /// `request` completes.
  std::vector<int> num_items_recv;
  /// Neighbourhood communicator. The caller must `MPI_Comm_free` it.
  MPI_Comm neigh_comm;
  /// Pending `MPI_Ineighbor_alltoall` for `num_items_recv`. The caller
  /// must `MPI_Wait` it before reading `num_items_recv`.
  MPI_Request request;
};

DistributionPlan compute_distribution_plan(
    MPI_Comm comm, const graph::AdjacencyList<std::int32_t>& destinations)
{
  DistributionPlan plan;

  // Build (dest, index, owning rank) list and sort
  plan.dest_to_index.reserve(destinations.array().size());
  for (std::int32_t i = 0; i < destinations.num_nodes(); ++i)
  {
    auto di = destinations.links(i);
    std::ranges::transform(di, std::back_inserter(plan.dest_to_index),
                           [i, d0 = di.front()](auto d) -> std::array<int, 3>
                           { return {d, i, d0}; });
  }

  // Only grouping by destination rank is required (order within a group
  // is irrelevant downstream), and the key is bounded by the
  // communicator size, so a radix sort keyed on the destination rank
  // alone is used rather than a full lexicographic sort.
  dolfinx::radix_sort(plan.dest_to_index, [](const auto& e) { return e[0]; });

  // Build list of unique dest ranks and count number of rows to send to
  // each dest (by neighbourhood rank)
  {
    auto it = plan.dest_to_index.begin();
    while (it != plan.dest_to_index.end())
    {
      // Store global rank and find iterator to next global rank
      plan.dest.push_back(it->front());
      auto it1 = std::find_if(it, plan.dest_to_index.end(),
                              [r = plan.dest.back()](auto& idx)
                              { return idx[0] != r; });

      // Store number of items for current rank
      plan.num_items_per_dest.push_back(std::ranges::distance(it, it1));

      // Advance iterator
      it = it1;
    }
  }

  // Determine source ranks. Sort ranks to make distribution
  // deterministic.
  plan.src = dolfinx::MPI::compute_graph_edges_nbx(comm, plan.dest);
  std::ranges::sort(plan.src);

  // Create neighbourhood communicator
  plan.dest.reserve(1);
  MPI_Dist_graph_create_adjacent(
      comm, plan.src.size(), plan.src.data(), MPI_UNWEIGHTED, plan.dest.size(),
      plan.dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &plan.neigh_comm);

  // Send number of rows to receivers
  plan.num_items_recv.resize(plan.src.size());
  plan.num_items_per_dest.reserve(1);
  plan.num_items_recv.reserve(1);
  MPI_Ineighbor_alltoall(plan.num_items_per_dest.data(), 1, MPI_INT,
                         plan.num_items_recv.data(), 1, MPI_INT,
                         plan.neigh_comm, &plan.request);

  // Compute send displacements
  plan.send_disp.resize(plan.num_items_per_dest.size() + 1, 0);
  std::partial_sum(plan.num_items_per_dest.begin(),
                   plan.num_items_per_dest.end(),
                   std::next(plan.send_disp.begin()));

  return plan;
}

/// @brief Pack the rows at positions `[i0, i1)` of `dest_to_index`
/// into the send buffer.
///
/// This code is thread-safe: a call writes only the send buffer rows
/// its own range covers.
///
/// @param[in] dest_to_index (destination rank, local index, owning
/// rank) triples, sorted by destination rank.
/// @param[in] i0 First position in `dest_to_index` to pack.
/// @param[in] i1 One past the last position in `dest_to_index` to pack.
/// @param[in] list Rows to distribute, flattened row-major.
/// @param[in] shape1 Number of entries per row of `list`.
/// @param[in] buffer_shape1 Number of entries per buffer row, i.e.
/// `shape1` plus the owning rank and the original global index.
/// @param[in] offset_global Global index of this rank's first row.
/// @param[out] send_buffer Packed rows.
void pack_send_rows(std::span<const std::array<int, 3>> dest_to_index,
                    std::size_t i0, std::size_t i1,
                    std::span<const std::int64_t> list, std::size_t shape1,
                    std::size_t buffer_shape1, std::int64_t offset_global,
                    std::span<std::int64_t> send_buffer)
{
  for (std::size_t i = i0; i < i1; ++i)
  {
    const std::array<int, 3>& dest_data = dest_to_index[i];
    const std::size_t pos = dest_data[1];
    std::span b(send_buffer.data() + i * buffer_shape1, buffer_shape1);
    std::copy_n(std::next(list.begin(), pos * shape1), shape1, b.begin());

    auto info = b.last(2);
    info[0] = dest_data[2];        // Owning rank
    info[1] = pos + offset_global; // Original global index
  }
}
//-----------------------------------------------------------------------------

/// @brief Count how many of the received rows `[q0, q1)` are owned by
/// this rank.
///
/// @param[in] recv_buffer Received rows, flattened row-major.
/// @param[in] buffer_shape1 Number of entries per received row.
/// @param[in] q0 First received row to count.
/// @param[in] q1 One past the last received row to count.
/// @param[in] rank This rank.
/// @param[out] count Number of rows in `[q0, q1)` owned by `rank`.
void count_owned_rows(std::span<const std::int64_t> recv_buffer,
                      std::size_t buffer_shape1, std::size_t q0, std::size_t q1,
                      int rank, std::int32_t& count)
{
  count = 0;
  for (std::size_t q = q0; q < q1; ++q)
  {
    if (recv_buffer[q * buffer_shape1 + buffer_shape1 - 2] == rank)
      ++count;
  }
}
//-----------------------------------------------------------------------------

/// @brief Unpack the received rows `[q0, q1)` into the output arrays.
///
/// Owned rows are placed before ghost rows, each group keeping the
/// order the rows were received in. A call writes the owned rows it
/// finds from `owned_pos` on and the ghost rows from `ghost_pos` on;
/// with those two positions set from the counts of the preceding
/// ranges (see ::count_owned_rows), the ranges write disjoint output
/// and together reproduce what one serial pass gives.
///
/// @param[in] recv_buffer Received rows, flattened row-major.
/// @param[in] buffer_shape1 Number of entries per received row.
/// @param[in] shape1 Number of entries per output row.
/// @param[in] q0 First received row to unpack.
/// @param[in] q1 One past the last received row to unpack.
/// @param[in] rank This rank.
/// @param[in] recv_disp Receive displacements, one per source rank.
/// @param[in] src Source ranks.
/// @param[in] owned_pos Output row for the first owned row found.
/// @param[in] ghost_pos Ghost row index for the first ghost row found.
/// @param[in] num_owned Total number of owned rows, i.e. the output
/// row at which the ghost rows start.
/// @param[out] data Unpacked rows, flattened row-major.
/// @param[out] global_indices Original global index of each row.
/// @param[out] src_ranks Source rank of each row.
/// @param[out] ghost_index_owner Owning rank of each ghost row.
void unpack_recv_rows(
    std::span<const std::int64_t> recv_buffer, std::size_t buffer_shape1,
    std::size_t shape1, std::size_t q0, std::size_t q1, int rank,
    std::span<const std::int32_t> recv_disp, std::span<const int> src,
    std::int32_t owned_pos, std::int32_t ghost_pos, std::int32_t num_owned,
    std::span<std::int64_t> data, std::span<std::int64_t> global_indices,
    std::span<int> src_ranks, std::span<int> ghost_index_owner)
{
  // Index of the source rank that sent row q0
  std::size_t p = 0;
  while (p + 1 < recv_disp.size() - 1
         and static_cast<std::size_t>(recv_disp[p + 1]) <= q0)
  {
    ++p;
  }

  for (std::size_t q = q0; q < q1; ++q)
  {
    while (static_cast<std::size_t>(recv_disp[p + 1]) <= q)
      ++p;

    std::span row(recv_buffer.data() + q * buffer_shape1, buffer_shape1);
    auto info = row.last(2);
    int owner = static_cast<int>(info[0]);
    std::int32_t pos;
    if (owner == rank)
      pos = owned_pos++;
    else
    {
      pos = num_owned + ghost_pos;
      ghost_index_owner[ghost_pos] = owner;
      ++ghost_pos;
    }

    std::copy_n(row.begin(), shape1, std::next(data.begin(), pos * shape1));
    global_indices[pos] = info[1];
    src_ranks[pos] = src[p];
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
bool graph::has_partitioner(const AnyPartitionFunction& partitioner)
{
  return std::visit([](const auto& p) { return static_cast<bool>(p); },
                    partitioner);
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t> graph::partition_graph(
    MPI_Comm comm, int nparts, const AdjacencyList<std::int64_t>& local_graph,
    std::optional<std::span<const std::int32_t>> node_weights,
    std::optional<std::span<const std::int32_t>> edge_weights, bool ghosting)
{
#if HAS_PARMETIS
  return graph::parmetis::partitioner()(comm, nparts, local_graph, node_weights,
                                        edge_weights, ghosting);
#elif HAS_PTSCOTCH
  return graph::scotch::partitioner()(comm, nparts, local_graph, node_weights,
                                      edge_weights, ghosting);
#elif HAS_KAHIP
  return graph::kahip::partitioner()(comm, nparts, local_graph, node_weights,
                                     edge_weights, ghosting);
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

  DistributionPlan plan = compute_distribution_plan(comm, destinations);

  // Pack send buffer
  std::vector<std::int64_t> send_buffer(buffer_shape1 * plan.send_disp.back(),
                                        -1);
  {
    assert(plan.send_disp.back() == (std::int32_t)plan.dest_to_index.size());
    for (std::size_t i = 0; i < plan.dest_to_index.size(); ++i)
    {
      std::array<int, 3> dest_data = plan.dest_to_index[i];
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
  MPI_Wait(&plan.request, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_disp(plan.num_items_recv.size() + 1, 0);
  std::partial_sum(plan.num_items_recv.begin(), plan.num_items_recv.end(),
                   std::next(recv_disp.begin()));

  // Send/receive data facet
  MPI_Datatype compound_type;
  MPI_Type_contiguous(buffer_shape1, MPI_INT64_T, &compound_type);
  MPI_Type_commit(&compound_type);
  std::vector<std::int64_t> recv_buffer(buffer_shape1 * recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer.data(), plan.num_items_per_dest.data(),
                         plan.send_disp.data(), compound_type,
                         recv_buffer.data(), plan.num_items_recv.data(),
                         recv_disp.data(), compound_type, plan.neigh_comm);
  MPI_Type_free(&compound_type);
  MPI_Comm_free(&plan.neigh_comm);

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
    const int src_rank = plan.src[p];
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
                         const graph::AdjacencyList<std::int32_t>& destinations,
                         int num_threads)
{
  if (num_threads < 1)
    throw std::invalid_argument("num_threads must be >= 1.");

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

  DistributionPlan plan = compute_distribution_plan(comm, destinations);

  // Pack send buffer. The buffer is allocated without initialisation
  // (and so via std::unique_ptr, not std::vector) because every entry
  // of every row is written below before it is sent -- the fill is GBs
  // of pure cost for a mesh with tens of millions of cells.
  assert(plan.send_disp.back() == (std::int32_t)plan.dest_to_index.size());
  const std::size_t send_size = buffer_shape1 * plan.send_disp.back();
  std::unique_ptr<std::int64_t[]> send_storage
      = std::make_unique_for_overwrite<std::int64_t[]>(send_size);
  std::span<std::int64_t> send_buffer(send_storage.get(), send_size);
  {
    std::vector<std::jthread> threads;
    for (int i = 1; i < num_threads; ++i)
    {
      auto [i0, i1]
          = common::local_range(i, plan.dest_to_index.size(), num_threads);
      threads.emplace_back(
          pack_send_rows,
          std::span<const std::array<int, 3>>(plan.dest_to_index), i0, i1, list,
          shape[1], buffer_shape1, offset_global, send_buffer);
    }
    auto [i0, i1]
        = common::local_range(0, plan.dest_to_index.size(), num_threads);
    pack_send_rows(plan.dest_to_index, i0, i1, list, shape[1], buffer_shape1,
                   offset_global, send_buffer);
  }

  // Prepare receive displacement
  MPI_Wait(&plan.request, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_disp(plan.num_items_recv.size() + 1, 0);
  std::partial_sum(plan.num_items_recv.begin(), plan.num_items_recv.end(),
                   std::next(recv_disp.begin()));

  // Send/receive data facet
  MPI_Datatype compound_type;
  MPI_Type_contiguous(buffer_shape1, MPI_INT64_T, &compound_type);
  MPI_Type_commit(&compound_type);
  const std::size_t recv_size = buffer_shape1 * recv_disp.back();
  std::unique_ptr<std::int64_t[]> recv_storage
      = std::make_unique_for_overwrite<std::int64_t[]>(recv_size);
  std::span<std::int64_t> recv_buffer(recv_storage.get(), recv_size);
  MPI_Neighbor_alltoallv(send_buffer.data(), plan.num_items_per_dest.data(),
                         plan.send_disp.data(), compound_type,
                         recv_buffer.data(), plan.num_items_recv.data(),
                         recv_disp.data(), compound_type, plan.neigh_comm);
  MPI_Type_free(&compound_type);
  MPI_Comm_free(&plan.neigh_comm);

  spdlog::debug("Received {} data on {} [{}]", recv_disp.back(), rank,
                shape[1]);

  // Unpack receive buffer. Owned rows are counted first so that the
  // output arrays can be sized and written exactly once, in place:
  // appending owned and ghost rows to separate vectors and joining
  // them afterwards instead copies (and, via shrink_to_fit,
  // reallocates) every row a second time.
  const std::size_t num_recv = recv_disp.back();
  std::vector<std::int32_t> num_owned_t(num_threads, 0);
  {
    std::vector<std::jthread> threads;
    for (int i = 1; i < num_threads; ++i)
    {
      auto [q0, q1] = common::local_range(i, num_recv, num_threads);
      threads.emplace_back(
          count_owned_rows, std::span<const std::int64_t>(recv_buffer),
          buffer_shape1, q0, q1, rank, std::ref(num_owned_t[i]));
    }
    auto [q0, q1] = common::local_range(0, num_recv, num_threads);
    count_owned_rows(recv_buffer, buffer_shape1, q0, q1, rank, num_owned_t[0]);
  }

  std::vector<std::int32_t> owned_offsets(num_threads + 1, 0);
  std::partial_sum(num_owned_t.begin(), num_owned_t.end(),
                   std::next(owned_offsets.begin()));
  const std::int32_t num_owned_rows = owned_offsets.back();

  // Each range starts writing owned rows at the number of owned rows
  // the ranges before it found, and ghost rows at the number of rows
  // before it that were not owned.
  std::vector<std::int64_t> data(num_recv * shape[1]);
  std::vector<std::int64_t> global_indices(num_recv);
  std::vector<int> src_ranks(num_recv);
  std::vector<int> ghost_index_owner(num_recv - num_owned_rows);
  {
    std::vector<std::jthread> threads;
    for (int i = 1; i < num_threads; ++i)
    {
      auto [q0, q1] = common::local_range(i, num_recv, num_threads);
      threads.emplace_back(
          unpack_recv_rows, std::span<const std::int64_t>(recv_buffer),
          buffer_shape1, shape[1], q0, q1, rank,
          std::span<const std::int32_t>(recv_disp),
          std::span<const int>(plan.src), owned_offsets[i],
          static_cast<std::int32_t>(q0) - owned_offsets[i], num_owned_rows,
          std::span<std::int64_t>(data),
          std::span<std::int64_t>(global_indices), std::span<int>(src_ranks),
          std::span<int>(ghost_index_owner));
    }
    auto [q0, q1] = common::local_range(0, num_recv, num_threads);
    unpack_recv_rows(
        recv_buffer, buffer_shape1, shape[1], q0, q1, rank, recv_disp, plan.src,
        owned_offsets[0], static_cast<std::int32_t>(q0) - owned_offsets[0],
        num_owned_rows, data, global_indices, src_ranks, ghost_index_owner);
  }

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
  in_edges.reserve(1);
  neighbors.reserve(1);
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
