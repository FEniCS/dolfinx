// Copyright (C) 2019-2026 Garth N. Wells, Chris Richardson and Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "partitioners.h"
#include <algorithm>
#include <boost/unordered/unordered_flat_set.hpp>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <format>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <vector>

#ifdef HAS_PTSCOTCH
extern "C"
{
#include <ptscotch.h>
}
#endif

#ifdef HAS_PARMETIS
extern "C"
{
#include <parmetis.h>
}
#endif

#ifdef HAS_KAHIP
#include <parhip_interface.h>
#endif

using namespace dolfinx;

template <typename T>
graph::AdjacencyList<int> dolfinx::graph::compute_destination_ranks(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& graph,
    const std::vector<T>& node_disp, const std::vector<T>& part)
{
  common::Timer timer("Extend graph destination ranks for halo");

  const int rank = dolfinx::MPI::rank(comm);
  const std::int64_t range0 = node_disp[rank];
  const std::int64_t range1 = node_disp[rank + 1];
  assert(static_cast<std::int32_t>(range1 - range0) == graph.num_nodes());

  // Wherever an owned 'node' goes, so must the nodes connected to it by
  // an edge ('node1'). Task is to let the owner of node1 know the extra
  // ranks that it needs to send node1 to.
  std::vector<std::array<std::int64_t, 3>> node_to_dest;
  node_to_dest.reserve(graph.array().size());
  for (int node0 = 0; node0 < graph.num_nodes(); ++node0)
  {
    // Wherever 'node' goes to, so must the attached 'node1'
    for (auto node1 : graph.links(node0))
    {
      if (node1 < range0 or node1 >= range1)
      {
        auto it = std::ranges::upper_bound(node_disp, node1);
        int remote_rank = std::ranges::distance(node_disp.begin(), it) - 1;
        node_to_dest.push_back(
            {remote_rank, node1, static_cast<std::int64_t>(part[node0])});
      }
      else
        node_to_dest.push_back(
            {rank, node1, static_cast<std::int64_t>(part[node0])});
    }
  }

  // De-duplicate exact (dest, node1, partition) triples with a single
  // hash-set pass (O(1)-average per insert, no sort needed for dedup).
  // Then sort only by the dest-rank column (0), which is all the
  // grouping below depends on, rather than by all 3 columns; a radix
  // sort on the flattened data is used for that single column, as
  // node_to_dest can have tens of millions of entries for a large
  // mesh.
  {
    boost::unordered_flat_set<std::array<std::int64_t, 3>> unique_set(
        node_to_dest.begin(), node_to_dest.end());
    node_to_dest.assign(unique_set.begin(), unique_set.end());

    std::span<const std::int64_t> flat(
        reinterpret_cast<const std::int64_t*>(node_to_dest.data()),
        3 * node_to_dest.size());
    std::vector<std::int32_t> perm
        = dolfinx::sort_by_perm<std::int64_t, 16>(flat, 3, 1);
    std::vector<std::array<std::int64_t, 3>> sorted(node_to_dest.size());
    for (std::size_t i = 0; i < perm.size(); ++i)
      sorted[i] = node_to_dest[perm[i]];
    node_to_dest = std::move(sorted);
  }

  // Build send data and buffer
  std::vector<int> dest, send_sizes;
  std::vector<std::int64_t> send_buffer;
  {
    auto it = node_to_dest.begin();
    while (it != node_to_dest.end())
    {
      // Current destination rank
      dest.push_back(it->front());

      // Find iterator to next destination rank and pack send data
      auto it1
          = std::find_if(it, node_to_dest.end(), [r0 = dest.back()](auto& idx)
                         { return idx[0] != r0; });
      send_sizes.push_back(2 * std::ranges::distance(it, it1));
      for (auto itx = it; itx != it1; ++itx)
      {
        send_buffer.push_back(itx->at(1));
        send_buffer.push_back(itx->at(2));
      }

      it = it1;
    }
  }

  // Prepare send displacements
  std::vector<int> send_disp(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_disp.begin()));

  // Discover src ranks. ParMETIS/KaHIP are not scalable (holding an
  // array of size equal to the comm size), so no extra harm in using
  // non-scalable neighbourhood detection (which might be faster for
  // small rank counts).
  const std::vector<int> src
      = dolfinx::MPI::compute_graph_edges_pcx(comm, dest);

  // Create neighbourhood communicator
  MPI_Comm neigh_comm;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm);

  // Determine receives sizes
  std::vector<int> recv_sizes(dest.size());
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                        MPI_INT, neigh_comm);

  // Prepare receive displacements
  std::vector<int> recv_disp(recv_sizes.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_disp.begin()));

  // Send/receive data
  std::vector<std::int64_t> recv_buffer(recv_disp.back());
  MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(),
                         send_disp.data(), MPI_INT64_T, recv_buffer.data(),
                         recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                         neigh_comm);
  MPI_Comm_free(&neigh_comm);

  // Prepare (local node index, destination rank) array. Add local data,
  // then add the received data, and the make unique.
  std::vector<std::array<int, 2>> local_node_to_dest;
  local_node_to_dest.reserve(2 * part.size() + 2 * recv_buffer.size());
  for (auto d : part)
  {
    local_node_to_dest.push_back(
        {static_cast<int>(local_node_to_dest.size()), static_cast<int>(d)});
  }
  for (std::size_t i = 0; i < recv_buffer.size(); i += 2)
  {
    std::int64_t idx = recv_buffer[i];
    int d = recv_buffer[i + 1];
    assert(idx >= range0 and idx < range1);
    std::int32_t idx_local = idx - range0;
    local_node_to_dest.push_back({idx_local, d});
  }

  // De-duplicate with a single hash-set pass, then sort only by the
  // local-node-index column (0) -- the grouping below depends on that
  // column alone. As above, a radix sort on the flattened data is used
  // for that single column -- this array is sized by the local node
  // count plus received halo entries, and so can also have millions of
  // entries for a large mesh.
  {
    boost::unordered_flat_set<std::array<int, 2>> unique_set(
        local_node_to_dest.begin(), local_node_to_dest.end());
    local_node_to_dest.assign(unique_set.begin(), unique_set.end());

    std::span<const int> flat(
        reinterpret_cast<const int*>(local_node_to_dest.data()),
        2 * local_node_to_dest.size());
    std::vector<std::int32_t> perm = dolfinx::sort_by_perm<int, 16>(flat, 2, 1);
    std::vector<std::array<int, 2>> sorted(local_node_to_dest.size());
    for (std::size_t i = 0; i < perm.size(); ++i)
      sorted[i] = local_node_to_dest[perm[i]];
    local_node_to_dest = std::move(sorted);
  }
  // Compute offsets
  std::vector<std::int32_t> offsets(graph.num_nodes() + 1, 0);
  {
    std::vector<std::int32_t> num_dests(graph.num_nodes(), 0);
    for (auto x : local_node_to_dest)
      ++num_dests[x[0]];
    std::partial_sum(num_dests.begin(), num_dests.end(),
                     std::next(offsets.begin()));
  }

  // Fill data array
  std::vector<int> data(offsets.back());
  {
    std::vector<std::int32_t> pos = offsets;
    for (auto [x0, x1] : local_node_to_dest)
      data[pos[x0]++] = x1;
  }

  graph::AdjacencyList<int> g(std::move(data), std::move(offsets));

  // Make sure the owning rank comes first for each node
  for (std::int32_t i = 0; i < g.num_nodes(); ++i)
  {
    auto d = g.links(i);
    auto it = std::find(d.begin(), d.end(), part[i]);
    assert(it != d.end());
    std::iter_swap(d.begin(), it);
  }

  return g;
}

/// @cond
template graph::AdjacencyList<int> dolfinx::graph::compute_destination_ranks(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& graph,
    const std::vector<int>& node_disp, const std::vector<int>& part);

template graph::AdjacencyList<int> dolfinx::graph::compute_destination_ranks(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& graph,
    const std::vector<unsigned long long>& node_disp,
    const std::vector<unsigned long long>& part);
/// @endcond

//-----------------------------------------------------------------------------
#ifdef HAS_PTSCOTCH
graph::partition_fn graph::scotch::partitioner(graph::scotch::strategy strategy,
                                               double imbalance, int seed)
{
  return [imbalance, strategy,
          seed](MPI_Comm comm, int nparts,
                const AdjacencyList<std::int64_t>& graph,
                std::optional<std::span<const std::int32_t>> node_weights,
                std::optional<std::span<const std::int32_t>> edge_weights,
                bool ghosting)
  {
    spdlog::info("Compute graph partition using PT-SCOTCH");
    common::Timer timer("Compute graph partition (SCOTCH)");

    std::int64_t offset_global = 0;
    const std::int64_t num_owned = graph.num_nodes();
    MPI_Request request_offset_scan;
    MPI_Iexscan(&num_owned, &offset_global, 1, MPI_INT64_T, MPI_SUM, comm,
                &request_offset_scan);

    // C-style array indexing
    constexpr SCOTCH_Num baseval = 0;

    // Copy  graph data to get the required type (SCOTCH_Num)
    std::vector<SCOTCH_Num> edgeloctab(graph.array().begin(),
                                       graph.array().end());
    std::vector<SCOTCH_Num> vertloctab(graph.offsets().begin(),
                                       graph.offsets().end());

    // Create SCOTCH graph and initialise
    SCOTCH_Dgraph dgrafdat;
    int err = SCOTCH_dgraphInit(&dgrafdat, comm);
    if (err != 0)
      throw std::runtime_error("Error initializing SCOTCH graph");

    // FIXME: If the nodes have weights but this rank has no nodes, then
    //        SCOTCH may deadlock since vload.data() will be nullptr on
    //        this rank but not null on all other ranks.
    // Handle node weights
    std::vector<SCOTCH_Num> vload;
    if (node_weights)
      vload.assign(node_weights->begin(), node_weights->end());

    // Handle edge weights
    std::vector<SCOTCH_Num> edload;
    if (edge_weights)
      edload.assign(edge_weights->begin(), edge_weights->end());

    // Set seed and reset SCOTCH random number generator to produce
    // deterministic partitions on repeated calls
    SCOTCH_randomSeed(seed);
    SCOTCH_randomReset();

    // Build SCOTCH distributed graph (SCOTCH is not const-correct, so
    // we throw away constness and trust SCOTCH)
    common::Timer timer1("SCOTCH: call SCOTCH_dgraphBuild");
    err = SCOTCH_dgraphBuild(
        &dgrafdat, baseval, graph.num_nodes(), graph.num_nodes(),
        vertloctab.data(), nullptr, vload.data(), nullptr, edgeloctab.size(),
        edgeloctab.size(), edgeloctab.data(), nullptr, edload.data());
    if (err != 0)
      throw std::runtime_error("Error building SCOTCH graph");
    timer1.stop();
    timer1.flush();

// Check graph data for consistency
#ifndef NDEBUG
    err = SCOTCH_dgraphCheck(&dgrafdat);
    if (err != 0)
      throw std::runtime_error("Consistency error in SCOTCH graph");
#endif

    // Initialise partitioning strategy
    SCOTCH_Strat strat;
    SCOTCH_stratInit(&strat);

    // Set SCOTCH strategy
    SCOTCH_Num strat_val;
    switch (strategy)
    {
    case strategy::none:
      strat_val = SCOTCH_STRATDEFAULT;
      break;
    case strategy::balance:
      strat_val = SCOTCH_STRATBALANCE;
      break;
    case strategy::quality:
      strat_val = SCOTCH_STRATQUALITY;
      break;
    case strategy::safety:
      strat_val = SCOTCH_STRATSAFETY;
      break;
    case strategy::speed:
      strat_val = SCOTCH_STRATSPEED;
      break;
    case strategy::scalability:
      strat_val = SCOTCH_STRATSCALABILITY;
      break;
    default:
      throw std::runtime_error("Unknown SCOTCH strategy");
    }
    err = SCOTCH_stratDgraphMapBuild(&strat, strat_val, nparts, nparts,
                                     imbalance);
    if (err != 0)
      throw std::runtime_error("Error calling SCOTCH_stratDgraphMapBuild");

    // Count number of 'ghost' edges, i.e. an edge to a cell that does
    // not belong to the caller. A single hash-set pass over the (much
    // larger) full edge array avoids materialising and sorting a
    // separate ghost-edges vector just to count distinct values.
    std::int32_t num_ghost_nodes = 0;
    {
      MPI_Wait(&request_offset_scan, MPI_STATUS_IGNORE);
      std::array<std::int64_t, 2> range
          = {offset_global, offset_global + num_owned};
      boost::unordered_flat_set<std::int64_t> ghost_nodes;
      for (std::int64_t e : graph.array())
        if (e < range[0] or e >= range[1])
          ghost_nodes.insert(e);
      num_ghost_nodes = ghost_nodes.size();
    }

    // Resize vector to hold node partition indices with enough extra
    // space for ghost node partition information too. When there are no
    // nodes, vertgstnbr may be zero, and at least one dummy location must
    // be created.
    const std::int32_t vertgstnbr = graph.num_nodes() + num_ghost_nodes;
    std::vector<SCOTCH_Num> node_partition(std::max(1, vertgstnbr), 0);

    // Partition the graph
    common::Timer timer2("SCOTCH: call SCOTCH_dgraphPart");
    err = SCOTCH_dgraphPart(&dgrafdat, nparts, &strat, node_partition.data());
    if (err != 0)
      throw std::runtime_error("Error during SCOTCH partitioning");
    timer2.stop();
    timer2.flush();

    // Data arrays for adjacency list, where the edges are the destination
    // ranks for each node
    std::vector<std::int32_t> dests;
    std::vector<std::int32_t> offsets(1, 0);
    if (ghosting)
    {
      // Exchange halo with node_partition data for ghosts
      common::Timer timer3("SCOTCH: call SCOTCH_dgraphHalo");
      err = SCOTCH_dgraphHalo(&dgrafdat, node_partition.data(),
                              dolfinx::MPI::mpi_t<SCOTCH_Num>);
      if (err != 0)
        throw std::runtime_error("Error during SCOTCH halo exchange");
      timer3.stop();
      timer3.flush();

      // Get SCOTCH's locally indexed graph
      common::Timer timer4("Get SCOTCH graph data");
      SCOTCH_Num* edge_ghost_tab;
      SCOTCH_dgraphData(&dgrafdat, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, &edge_ghost_tab, nullptr, &comm);
      timer4.stop();
      timer4.flush();

      // Iterate through SCOTCH's local compact graph to find partition
      // boundaries and save to map
      common::Timer timer5("Extract partition boundaries from SCOTCH graph");

      // Create a map of local nodes to their additional destination
      // processes, due to ghosting
      std::map<std::int32_t, std::set<std::int32_t>> local_node_to_dests;
      for (std::int32_t node0 = 0; node0 < graph.num_nodes(); ++node0)
      {
        // Get all edges outward from node i
        const std::int32_t node0_rank = node_partition[node0];
        for (SCOTCH_Num j = vertloctab[node0]; j < vertloctab[node0 + 1]; ++j)
        {
          // Any edge which connects to a different partition will be a
          // ghost
          const std::int32_t node1_rank = node_partition[edge_ghost_tab[j]];
          if (node0_rank != node1_rank)
            local_node_to_dests[node0].insert(node1_rank);
        }
      }
      timer5.stop();
      timer5.flush();

      offsets.reserve(graph.num_nodes() + 1);
      for (std::int32_t i = 0; i < graph.num_nodes(); ++i)
      {
        dests.push_back(node_partition[i]);
        if (auto it = local_node_to_dests.find(i);
            it != local_node_to_dests.end())
        {
          dests.insert(dests.end(), it->second.begin(), it->second.end());
        }

        offsets.push_back(dests.size());
      }

      dests.shrink_to_fit();
    }
    else
    {
      offsets.resize(graph.num_nodes() + 1);
      std::iota(offsets.begin(), offsets.end(), 0);
      dests = std::vector<std::int32_t>(
          node_partition.begin(),
          std::next(node_partition.begin(), graph.num_nodes()));
    }

    // Clean up SCOTCH objects
    SCOTCH_dgraphExit(&dgrafdat);
    SCOTCH_stratExit(&strat);

    return graph::AdjacencyList(std::move(dests), std::move(offsets));
  };
}
#endif
//-----------------------------------------------------------------------------
#ifdef HAS_PARMETIS
namespace
{
/// @brief Split `comm` into the ranks holding graph nodes, and build
/// the node displacement array for the resulting sub-communicator.
///
/// ParMETIS fails (crashes) if a rank does not have any graph data,
/// so partitioning must happen only on ranks that have data.
///
/// @param[in] comm Communicator to split.
/// @param[in] num_local_nodes Number of graph nodes on this rank.
/// @return Sub-communicator holding the ranks with `num_local_nodes >
/// 0` (`MPI_COMM_NULL` on the other ranks), and that sub-communicator's
/// node displacement array (empty on the other ranks). The caller must
/// free the returned communicator with `MPI_Comm_free` once done with
/// it, if it is not `MPI_COMM_NULL`.
std::pair<MPI_Comm, std::vector<idx_t>>
split_and_build_node_disp(MPI_Comm comm, idx_t num_local_nodes)
{
  const int rank = dolfinx::MPI::rank(comm);
  const int color = num_local_nodes > 0 ? 1 : MPI_UNDEFINED;
  MPI_Comm pcomm = MPI_COMM_NULL;
  int ierr = MPI_Comm_split(comm, color, rank, &pcomm);
  dolfinx::MPI::check_error(comm, ierr);

  std::vector<idx_t> node_disp;
  if (pcomm != MPI_COMM_NULL)
  {
    const int psize = dolfinx::MPI::size(pcomm);
    node_disp = std::vector<idx_t>(psize + 1, 0);
    MPI_Allgather(&num_local_nodes, 1, dolfinx::MPI::mpi_t<idx_t>,
                  node_disp.data() + 1, 1, dolfinx::MPI::mpi_t<idx_t>, pcomm);
    std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
  }

  return {pcomm, std::move(node_disp)};
}

/// @brief Finalise a ParMETIS partition result: extend `part` to
/// per-node destination ranks with ghosts if `ghosting` is requested,
/// and free `pcomm`.
///
/// @param[in] pcomm Sub-communicator returned by
/// split_and_build_node_disp.
/// @param[in] ghosting Whether to compute ghost destinations.
/// @param[in] graph Local graph, used to compute ghost destinations
/// when `ghosting` is true.
/// @param[in] node_disp Node displacement array for `pcomm`.
/// @param[in] part Partition index of each local node.
/// @return Destination rank(s) for each node.
// FIXME: Is it implicit that the first entry is the owner?
graph::AdjacencyList<int>
finalise_partition(MPI_Comm pcomm, bool ghosting,
                   const graph::AdjacencyList<std::int64_t>& graph,
                   const std::vector<idx_t>& node_disp,
                   const std::vector<idx_t>& part)
{
  graph::AdjacencyList<int> dest
      = (ghosting and pcomm != MPI_COMM_NULL)
            ? graph::compute_destination_ranks(pcomm, graph, node_disp, part)
            : graph::regular_adjacency_list(
                  std::vector<int>(part.begin(), part.end()), 1);
  if (pcomm != MPI_COMM_NULL)
    MPI_Comm_free(&pcomm);
  return dest;
}

/// @brief ParMETIS element/edge weight vectors and the `wgtflag` value
/// that tells ParMETIS which of them are present.
struct ParmetisWeights
{
  idx_t wgtflag = 0;
  std::vector<idx_t> elmwgt;
  std::vector<idx_t> edgwgt;
};

/// @brief Build the ParMETIS element/edge weight vectors and `wgtflag`
/// from optional node/edge weight spans, logging which (if either) are
/// applied.
ParmetisWeights build_parmetis_weights(
    std::optional<std::span<const std::int32_t>> node_weights,
    std::optional<std::span<const std::int32_t>> edge_weights)
{
  ParmetisWeights w;
  if (node_weights)
    w.elmwgt.assign(node_weights->begin(), node_weights->end());
  if (edge_weights)
    w.edgwgt.assign(edge_weights->begin(), edge_weights->end());

  if (!w.elmwgt.empty())
  {
    spdlog::info("ParMETIS: applying node weights");
    w.wgtflag += 2;
  }
  if (!w.edgwgt.empty())
  {
    spdlog::info("ParMETIS: applying edge weights");
    w.wgtflag += 1;
  }

  return w;
}

} // namespace

graph::partition_fn graph::parmetis::partitioner(double imbalance,
                                                 std::array<int, 3> options)
{
  return [imbalance,
          options](MPI_Comm comm, idx_t nparts,
                   const graph::AdjacencyList<std::int64_t>& graph,
                   std::optional<std::span<const std::int32_t>> node_weights,
                   std::optional<std::span<const std::int32_t>> edge_weights,
                   bool ghosting)
  {
    spdlog::info("Compute graph partition using ParMETIS");
    common::Timer timer("Compute graph partition (ParMETIS)");

    if (nparts == 1 and dolfinx::MPI::size(comm) == 1)
    {
      // Nothing to be partitioned
      return regular_adjacency_list(
          std::vector<std::int32_t>(graph.num_nodes(), 0), 1);
    }

    auto [pcomm, node_disp]
        = split_and_build_node_disp(comm, graph.num_nodes());

    std::vector<idx_t> part(graph.num_nodes());
    if (pcomm != MPI_COMM_NULL)
    {
      std::vector<idx_t> array(graph.array().begin(), graph.array().end());
      std::vector<idx_t> offsets(graph.offsets().begin(),
                                 graph.offsets().end());

      // Options and data for ParMETIS
      std::array<idx_t, 3> opts = {options[0], options[1], options[2]};
      idx_t ncon = 1;
      idx_t edgecut(0), numflag(0);
      ParmetisWeights w = build_parmetis_weights(node_weights, edge_weights);

      std::vector<real_t> tpwgts(ncon * nparts,
                                 1.0 / static_cast<real_t>(nparts));
      real_t ubvec = static_cast<real_t>(imbalance);

      // Partition
      common::Timer timer1("ParMETIS: call ParMETIS_V3_PartKway");
      int err = ParMETIS_V3_PartKway(
          node_disp.data(), offsets.data(), array.data(), w.elmwgt.data(),
          w.edgwgt.data(), &w.wgtflag, &numflag, &ncon, &nparts, tpwgts.data(),
          &ubvec, opts.data(), &edgecut, part.data(), &pcomm);
      if (err != METIS_OK)
      {
        throw std::runtime_error(
            std::format("ParMETIS_V3_PartKway failed. Error code: {}", err));
      }
    }

    return finalise_partition(pcomm, ghosting, graph, node_disp, part);
  };
}
//-----------------------------------------------------------------------------
graph::partition_fn graph::parmetis::repartitioner(double ipc2redist,
                                                   double imbalance,
                                                   std::array<int, 3> options)
{
  return [ipc2redist, imbalance,
          options](MPI_Comm comm, idx_t nparts,
                   const graph::AdjacencyList<std::int64_t>& graph,
                   std::optional<std::span<const std::int32_t>> node_weights,
                   std::optional<std::span<const std::int32_t>> edge_weights,
                   bool ghosting)
  {
    spdlog::info("Compute graph re-partition using ParMETIS");
    common::Timer timer("Compute graph re-partition (ParMETIS)");

    const int rank = dolfinx::MPI::rank(comm);
    if (nparts != dolfinx::MPI::size(comm))
    {
      throw std::runtime_error(
          "Number of parts must equal the communicator size for "
          "re-partitioning, as the current partition is taken to be the "
          "current data distribution.");
    }

    if (nparts == 1)
    {
      // Nothing to be re-partitioned
      return regular_adjacency_list(
          std::vector<std::int32_t>(graph.num_nodes(), 0), 1);
    }

    auto [pcomm, node_disp]
        = split_and_build_node_disp(comm, graph.num_nodes());

    // The current partition is the current distribution, i.e. the nodes
    // held by this rank are currently assigned to this rank. ParMETIS
    // overwrites `part` with the new partition.
    std::vector<idx_t> part(graph.num_nodes(), rank);
    if (pcomm != MPI_COMM_NULL)
    {
      std::vector<idx_t> array(graph.array().begin(), graph.array().end());
      std::vector<idx_t> offsets(graph.offsets().begin(),
                                 graph.offsets().end());

      // Cost of moving each node between ranks, taken to be uniform
      std::vector<idx_t> vsize(std::max<std::size_t>(part.size(), 1), 1);

      // The current partition is supplied in `part`, so the partition and
      // the process layout are 'uncoupled' in ParMETIS terms
      std::array<idx_t, 4> opts
          = {options[0], options[1], options[2], PARMETIS_PSR_UNCOUPLED};
      idx_t ncon = 1;
      idx_t numflag(0), edgecut(0);
      ParmetisWeights w = build_parmetis_weights(node_weights, edge_weights);

      std::vector<real_t> tpwgts(ncon * nparts,
                                 1.0 / static_cast<real_t>(nparts));
      std::vector<real_t> ubvec(ncon, static_cast<real_t>(imbalance));
      real_t itr = static_cast<real_t>(ipc2redist);

      common::Timer timer1("ParMETIS: call ParMETIS_V3_AdaptiveRepart");
      int err = ParMETIS_V3_AdaptiveRepart(
          node_disp.data(), offsets.data(), array.data(), w.elmwgt.data(),
          vsize.data(), w.edgwgt.data(), &w.wgtflag, &numflag, &ncon, &nparts,
          tpwgts.data(), ubvec.data(), &itr, opts.data(), &edgecut, part.data(),
          &pcomm);
      if (err != METIS_OK)
      {
        throw std::runtime_error(std::format(
            "ParMETIS_V3_AdaptiveRepart failed. Error code: {}", err));
      }
    }

    return finalise_partition(pcomm, ghosting, graph, node_disp, part);
  };
}
//-----------------------------------------------------------------------------
std::vector<int> graph::parmetis::geom_partitioner(
    MPI_Comm comm, int nparts, std::span<const double> x, int gdim,
    std::optional<std::span<const std::int32_t>> node_weights)
{
  if (node_weights)
  {
    throw std::runtime_error(
        "ParMETIS_V3_PartGeom does not support node weights.");
  }

  spdlog::info("Compute geometric graph partition using ParMETIS");
  common::Timer timer("Compute graph partition (ParMETIS geometric)");

  const idx_t num_nodes = x.size() / gdim;

  if (nparts == 1 and dolfinx::MPI::size(comm) == 1)
  {
    // Nothing to be partitioned
    return std::vector<int>(num_nodes, 0);
  }

  auto [pcomm, node_disp] = split_and_build_node_disp(comm, num_nodes);

  std::vector<idx_t> part(num_nodes);
  if (pcomm != MPI_COMM_NULL)
  {
    if (nparts != dolfinx::MPI::size(pcomm))
    {
      throw std::runtime_error(
          "ParMETIS_V3_PartGeom partitions into one part per MPI rank, so "
          "the number of parts must equal the communicator size.");
    }

    // ParMETIS requires its own scalar type for the coordinates
    std::vector<real_t> xyz(x.begin(), x.end());
    idx_t ndims = gdim;

    common::Timer timer1("ParMETIS: call ParMETIS_V3_PartGeom");
    int err = ParMETIS_V3_PartGeom(node_disp.data(), &ndims, xyz.data(),
                                   part.data(), &pcomm);
    if (err != METIS_OK)
    {
      throw std::runtime_error(
          std::format("ParMETIS_V3_PartGeom failed. Error code: {}", err));
    }

    MPI_Comm_free(&pcomm);
  }

  return std::vector<int>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
graph::hybrid_partition_fn
graph::parmetis::geom_partitioner_kway(double imbalance,
                                       std::array<int, 3> options)
{
  return [imbalance,
          options](MPI_Comm comm, idx_t nparts,
                   const graph::AdjacencyList<std::int64_t>& graph,
                   std::span<const double> x,
                   std::optional<std::span<const std::int32_t>> node_weights,
                   std::optional<std::span<const std::int32_t>> edge_weights,
                   bool ghosting)
  {
    spdlog::info("Compute geometric graph partition using ParMETIS");
    common::Timer timer("Compute graph partition (ParMETIS geometric)");

    const idx_t num_nodes = graph.num_nodes();
    const idx_t gdim
        = num_nodes > 0 ? static_cast<idx_t>(x.size()) / num_nodes : 0;
    if (static_cast<std::int64_t>(x.size())
        != static_cast<std::int64_t>(gdim) * num_nodes)
    {
      throw std::runtime_error(
          "Number of coordinates does not match number of graph nodes.");
    }

    if (nparts == 1 and dolfinx::MPI::size(comm) == 1)
    {
      // Nothing to be partitioned
      return regular_adjacency_list(std::vector<std::int32_t>(num_nodes, 0), 1);
    }

    auto [pcomm, node_disp] = split_and_build_node_disp(comm, num_nodes);

    std::vector<idx_t> part(num_nodes);
    if (pcomm != MPI_COMM_NULL)
    {
      // ParMETIS requires its own scalar type for the coordinates
      std::vector<real_t> xyz(x.begin(), x.end());
      idx_t ndims = gdim;

      std::vector<idx_t> array(graph.array().begin(), graph.array().end());
      std::vector<idx_t> offsets(graph.offsets().begin(),
                                 graph.offsets().end());
      std::array<idx_t, 3> opts = {options[0], options[1], options[2]};
      idx_t ncon = 1;
      idx_t edgecut(0), numflag(0);
      ParmetisWeights w = build_parmetis_weights(node_weights, edge_weights);

      std::vector<real_t> tpwgts(ncon * nparts,
                                 1.0 / static_cast<real_t>(nparts));
      real_t ubvec = static_cast<real_t>(imbalance);

      common::Timer timer1("ParMETIS: call ParMETIS_V3_PartGeomKway");
      int err = ParMETIS_V3_PartGeomKway(
          node_disp.data(), offsets.data(), array.data(), w.elmwgt.data(),
          w.edgwgt.data(), &w.wgtflag, &numflag, &ndims, xyz.data(), &ncon,
          &nparts, tpwgts.data(), &ubvec, opts.data(), &edgecut, part.data(),
          &pcomm);
      if (err != METIS_OK)
      {
        throw std::runtime_error(std::format(
            "ParMETIS_V3_PartGeomKway failed. Error code: {}", err));
      }
    }

    return finalise_partition(pcomm, ghosting, graph, node_disp, part);
  };
}
//-----------------------------------------------------------------------------
#endif

#ifdef HAS_KAHIP

//----------------------------------------------------------------------------
graph::partition_fn graph::kahip::partitioner(int mode, int seed,
                                              double imbalance,
                                              bool suppress_output)
{
  return [mode, seed, imbalance, suppress_output](
             MPI_Comm comm, int nparts,
             const graph::AdjacencyList<std::int64_t>& graph,
             std::optional<std::span<const std::int32_t>> node_weights,
             std::optional<std::span<const std::int32_t>> edge_weights,
             bool ghosting)
  {
    spdlog::info("Compute graph partition using (parallel) KaHIP");

    // KaHIP integer type
    using T = unsigned long long;

    common::Timer timer("Compute graph partition (KaHIP)");

    std::vector<T> vwgt;
    if (node_weights)
      vwgt.assign(node_weights->begin(), node_weights->end());
    std::vector<T> adjcwgt;
    if (edge_weights)
      adjcwgt.assign(edge_weights->begin(), edge_weights->end());

    // Build adjacency list data
    common::Timer timer1("KaHIP: build adjacency data");
    std::vector<T> node_disp(dolfinx::MPI::size(comm) + 1, 0);
    const T num_local_nodes = graph.num_nodes();

    // KaHIP internally relies on an unsigned long long int type, which is not
    // easily convertible to a general mpi type due to platform specific
    // differences. So we can not rely on the general mpi_t<> mapping and do it
    // by hand in this sole occurrence.
    MPI_Allgather(&num_local_nodes, 1, MPI_UNSIGNED_LONG_LONG,
                  node_disp.data() + 1, 1, MPI_UNSIGNED_LONG_LONG, comm);
    std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
    std::vector<T> array(graph.array().begin(), graph.array().end());
    std::vector<T> offsets(graph.offsets().begin(), graph.offsets().end());
    timer1.stop();

    // Call KaHIP to partition graph
    common::Timer timer2("KaHIP: call ParHIPPartitionKWay");
    std::vector<T> part(graph.num_nodes());
    int edgecut = 0;
    double _imbalance = imbalance;
    ParHIPPartitionKWay(node_disp.data(), offsets.data(), array.data(),
                        vwgt.data(), adjcwgt.data(), &nparts, &_imbalance,
                        suppress_output, seed, mode, &edgecut, part.data(),
                        &comm);
    timer2.stop();

    if (ghosting)
      return graph::compute_destination_ranks(comm, graph, node_disp, part);
    else
    {
      return regular_adjacency_list(std::vector<int>(part.begin(), part.end()),
                                    1);
    }
  };
}
//----------------------------------------------------------------------------
#endif
