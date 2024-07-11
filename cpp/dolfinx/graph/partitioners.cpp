// Copyright (C) 2019-2020 Garth N. Wells, Chris Richardson and Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "partitioners.h"
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <map>
#include <numeric>
#include <set>
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

namespace
{
/// @todo Is it un-documented that the owning rank must come first in
/// reach list of edges?
///
/// @param[in] comm The communicator
/// @param[in] graph Graph, using global indices for graph edges
/// @param[in] node_disp The distribution of graph nodes across MPI
/// ranks. The global index `gidx` of local index `lidx` is `lidx +
/// node_disp[my_rank]`.
/// @param[in] part The destination rank for owned nodes, i.e. `dest[i]`
/// is the destination of the node with local index `i`.
/// @return Destination ranks for each local node.
template <typename T>
graph::AdjacencyList<int> compute_destination_ranks(
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
  for (int node0 = 0; node0 < graph.num_nodes(); ++node0)
  {
    // Wherever 'node' goes to, so must the attached 'node1'
    for (auto node1 : graph.links(node0))
    {
      if (node1 < range0 or node1 >= range1)
      {
        auto it = std::upper_bound(node_disp.begin(), node_disp.end(), node1);
        int remote_rank = std::distance(node_disp.begin(), it) - 1;
        node_to_dest.push_back(
            {remote_rank, node1, static_cast<std::int64_t>(part[node0])});
      }
      else
        node_to_dest.push_back(
            {rank, node1, static_cast<std::int64_t>(part[node0])});
    }
  }
  std::ranges::sort(node_to_dest);
  node_to_dest.erase(std::unique(node_to_dest.begin(), node_to_dest.end()),
                     node_to_dest.end());

  // Build send data and buffer
  std::vector<int> dest, send_sizes;
  std::vector<std::int64_t> send_buffer;
  {
    auto it = node_to_dest.begin();
    while (it != node_to_dest.end())
    {
      // Current destination rank
      dest.push_back((*it)[0]);

      // Find iterator to next destination rank and pack send data
      auto it1
          = std::find_if(it, node_to_dest.end(), [r0 = dest.back()](auto& idx)
                         { return idx[0] != r0; });
      send_sizes.push_back(2 * std::distance(it, it1));
      for (auto itx = it; itx != it1; ++itx)
      {
        send_buffer.push_back((*itx)[1]);
        send_buffer.push_back((*itx)[2]);
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
  std::ranges::sort(local_node_to_dest);
  local_node_to_dest.erase(
      std::unique(local_node_to_dest.begin(), local_node_to_dest.end()),
      local_node_to_dest.end());

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
//-----------------------------------------------------------------------------
#ifdef HAS_PARMETIS
template <typename T>
std::vector<int> adaptive_repartition(MPI_Comm comm,
                                      const graph::AdjacencyList<T>& adj_graph,
                                      double weight)
{
  common::Timer timer(
      "Compute graph partition (ParMETIS Adaptive Repartition)");

  // Options for ParMETIS
  idx_t options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  options[3] = PARMETIS_PSR_UNCOUPLED;
  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all
  // migration if already balanced.  Try PARMETIS_PSR_UNCOUPLED for
  // better edge cut.

  common::Timer timer1("ParMETIS: call ParMETIS_V3_AdaptiveRepart");
  real_t _itr = weight;
  std::vector<idx_t> part(adj_graph.num_nodes());
  std::vector<idx_t> vsize(part.size(), 1);
  assert(!part.empty());

  // Number of partitions (one for each process)
  idx_t nparts = dolfinx::MPI::size(comm);

  // Remaining ParMETIS parameters
  idx_t ncon = 1;
  idx_t* elmwgt = nullptr;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon * nparts, 1.0 / static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to repartition graph
  int err = ParMETIS_V3_AdaptiveRepart(
      adj_graph.node_distribution().data(), adj_graph.nodes().data(),
      adj_graph.edges().data(), elmwgt, nullptr, vsize.data(), &wgtflag,
      &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(), &_itr, options,
      &edgecut, part.data(), &comm);
  assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data and return
  return std::vector<int>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
template <typename T>
std::vector<int> refine(MPI_Comm comm, const graph::AdjacencyList<T>& adj_graph)
{
  common::Timer timer("Compute graph partition (ParMETIS Refine)");

  // Get some MPI data
  const int process_number = dolfinx::MPI::rank(comm);

  // Options for ParMETIS
  idx_t options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  // options[3] = PARMETIS_PSR_UNCOUPLED;

  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all
  // migration if already balanced.  Try PARMETIS_PSR_UNCOUPLED for
  // better edge cut.

  // Partitioning array to be computed by ParMETIS. Prefill with
  // process_number.
  const std::int32_t num_local_cells = adj_graph.num_nodes();
  std::vector<idx_t> part(num_local_cells, process_number);
  assert(!part.empty());

  // Number of partitions (one for each process)
  idx_t nparts = dolfinx::MPI::size(comm);
  // Remaining ParMETIS parameters
  idx_t ncon = 1;
  idx_t* elmwgt = nullptr;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon * nparts, 1.0 / static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to partition graph
  common::Timer timer1("ParMETIS: call ParMETIS_V3_RefineKway");
  int err = ParMETIS_V3_RefineKway(
      adj_graph.node_distribution().data(), adj_graph.nodes().data(),
      adj_graph.edges().data(), elmwgt, nullptr, &wgtflag, &numflag, &ncon,
      &nparts, tpwgts.data(), ubvec.data(), options, &edgecut, part.data(),
      &comm);
  assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data
  return std::vector<int>(part.begin(), part.end());
  //-----------------------------------------------------------------------------
}
#endif
} // namespace

//-----------------------------------------------------------------------------
#ifdef HAS_PTSCOTCH
graph::partition_fn graph::scotch::partitioner(graph::scotch::strategy strategy,
                                               double imbalance, int seed)
{
  return [imbalance, strategy, seed](MPI_Comm comm, int nparts,
                                     const AdjacencyList<std::int64_t>& graph,
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
    // Handle node weights (disabled for now)
    std::vector<SCOTCH_Num> node_weights;
    std::vector<SCOTCH_Num> vload;
    if (!node_weights.empty())
      vload.assign(node_weights.begin(), node_weights.end());

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
        edgeloctab.size(), edgeloctab.data(), nullptr, nullptr);
    if (err != 0)
      throw std::runtime_error("Error building SCOTCH graph");
    timer1.stop();

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
    // not belong to the caller
    std::int32_t num_ghost_nodes = 0;
    {
      MPI_Wait(&request_offset_scan, MPI_STATUS_IGNORE);
      std::array<std::int64_t, 2> range
          = {offset_global, offset_global + num_owned};
      std::vector<std::int64_t> ghost_edges;
      std::copy_if(graph.array().begin(), graph.array().end(),
                   std::back_inserter(ghost_edges),
                   [range](auto e) { return e < range[0] or e >= range[1]; });
      std::ranges::sort(ghost_edges);
      auto it = std::unique(ghost_edges.begin(), ghost_edges.end());
      num_ghost_nodes = std::distance(ghost_edges.begin(), it);
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

    // Data arrays for adjacency list, where the edges are the destination
    // ranks for each node
    std::vector<std::int32_t> dests;
    std::vector<std::int32_t> offsets(1, 0);
    if (ghosting)
    {
      // Exchange halo with node_partition data for ghosts
      common::Timer timer3("SCOTCH: call SCOTCH_dgraphHalo");
      err = SCOTCH_dgraphHalo(&dgrafdat, node_partition.data(),
                              dolfinx::MPI::mpi_type<SCOTCH_Num>());
      if (err != 0)
        throw std::runtime_error("Error during SCOTCH halo exchange");
      timer3.stop();

      // Get SCOTCH's locally indexed graph
      common::Timer timer4("Get SCOTCH graph data");
      SCOTCH_Num* edge_ghost_tab;
      SCOTCH_dgraphData(&dgrafdat, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, &edge_ghost_tab, nullptr, &comm);
      timer4.stop();

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
graph::partition_fn graph::parmetis::partitioner(double imbalance,
                                                 std::array<int, 3> options)
{
  return [imbalance, options](MPI_Comm comm, idx_t nparts,
                              const graph::AdjacencyList<std::int64_t>& graph,
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

    // Note: ParMETIS fails (crashes) if a rank does not have any graph
    // data. Therefore we split the communicator such that ParMETIS
    // partitioning happens only on ranks that have data. Ideallt we
    // wouldn't need to do this.
    constexpr bool split_comm = true;
    MPI_Comm pcomm = MPI_COMM_NULL;
    if (split_comm)
    {
      int rank = dolfinx::MPI::rank(comm);
      int color = graph.num_nodes() > 0 ? 1 : MPI_UNDEFINED;
      int ierr = MPI_Comm_split(comm, color, rank, &pcomm);
      dolfinx::MPI::check_error(comm, ierr);
    }
    else
      pcomm = comm;

    std::vector<idx_t> part(graph.num_nodes());
    std::vector<idx_t> node_disp;
    if (pcomm != MPI_COMM_NULL)
    {
      // Build adjacency list data
      const int psize = dolfinx::MPI::size(pcomm);
      const idx_t num_local_nodes = graph.num_nodes();
      node_disp = std::vector<idx_t>(psize + 1, 0);
      MPI_Allgather(&num_local_nodes, 1, dolfinx::MPI::mpi_type<idx_t>(),
                    node_disp.data() + 1, 1, dolfinx::MPI::mpi_type<idx_t>(),
                    pcomm);
      std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
      std::vector<idx_t> array(graph.array().begin(), graph.array().end());
      std::vector<idx_t> offsets(graph.offsets().begin(),
                                 graph.offsets().end());

      // Options and data for ParMETIS
      std::array<idx_t, 3> opts = {options[0], options[1], options[2]};
      idx_t ncon = 1;
      idx_t* elmwgt = nullptr;
      idx_t wgtflag(0), edgecut(0), numflag(0);
      std::vector<real_t> tpwgts(ncon * nparts,
                                 1.0 / static_cast<real_t>(nparts));
      real_t ubvec = static_cast<real_t>(imbalance);

      // Partition
      common::Timer timer1("ParMETIS: call ParMETIS_V3_PartKway");
      int err = ParMETIS_V3_PartKway(
          node_disp.data(), offsets.data(), array.data(), elmwgt, nullptr,
          &wgtflag, &numflag, &ncon, &nparts, tpwgts.data(), &ubvec,
          opts.data(), &edgecut, part.data(), &pcomm);
      if (err != METIS_OK)
      {
        throw std::runtime_error("ParMETIS_V3_PartKway failed. Error code: "
                                 + std::to_string(err));
      }
    }

    if (ghosting and pcomm != MPI_COMM_NULL)
    {
      // FIXME: Is it implicit that the first entry is the owner?
      graph::AdjacencyList<int> dest
          = compute_destination_ranks(pcomm, graph, node_disp, part);
      if (split_comm)
        MPI_Comm_free(&pcomm);
      return dest;
    }
    else
    {
      if (split_comm and pcomm != MPI_COMM_NULL)
        MPI_Comm_free(&pcomm);
      return regular_adjacency_list(std::vector<int>(part.begin(), part.end()),
                                    1);
    }
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
             const graph::AdjacencyList<std::int64_t>& graph, bool ghosting)
  {
    spdlog::info("Compute graph partition using (parallel) KaHIP");

    // KaHIP integer type
    using T = unsigned long long;

    common::Timer timer("Compute graph partition (KaHIP)");

    // Graph does not have vertex or adjacency weights, so we use null
    // pointers as arguments
    T *vwgt(nullptr), *adjcwgt(nullptr);

    // Build adjacency list data
    common::Timer timer1("KaHIP: build adjacency data");
    std::vector<T> node_disp(dolfinx::MPI::size(comm) + 1, 0);
    const T num_local_nodes = graph.num_nodes();
    MPI_Allgather(&num_local_nodes, 1, dolfinx::MPI::mpi_type<T>(),
                  node_disp.data() + 1, 1, dolfinx::MPI::mpi_type<T>(), comm);
    std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
    std::vector<T> array(graph.array().begin(), graph.array().end());
    std::vector<T> offsets(graph.offsets().begin(), graph.offsets().end());
    timer1.stop();

    // Call KaHIP to partition graph
    common::Timer timer2("KaHIP: call ParHIPPartitionKWay");
    std::vector<T> part(graph.num_nodes());
    int edgecut = 0;
    double _imbalance = imbalance;
    ParHIPPartitionKWay(node_disp.data(), offsets.data(), array.data(), vwgt,
                        adjcwgt, &nparts, &_imbalance, suppress_output, seed,
                        mode, &edgecut, part.data(), &comm);
    timer2.stop();

    if (ghosting)
      return compute_destination_ranks(comm, graph, node_disp, part);
    else
    {
      return regular_adjacency_list(std::vector<int>(part.begin(), part.end()),
                                    1);
    }
  };
}
//----------------------------------------------------------------------------
#endif
