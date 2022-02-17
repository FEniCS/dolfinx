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
//-----------------------------------------------------------------------------
template <typename T>
graph::AdjacencyList<std::int32_t> compute_destination_ranks(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& graph,
    const std::vector<T>& node_disp, const std::vector<T>& part)
{
  // Create a map from local nodes to their additional destination
  // ranks, due to ghosting.

  // Work out halo nodes for current division of graph
  common::Timer timer2("Compute graph halo data (ParMETIS/KaHIP)");

  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);
  const std::int64_t range0 = node_disp[rank];
  const std::int64_t range1 = node_disp[rank + 1];
  assert(static_cast<std::int32_t>(range1 - range0) == graph.num_nodes());

  // local indexing "i"
  std::map<std::int32_t, std::set<int>> halo_node_to_remotes;
  for (int node = 0; node < graph.num_nodes(); ++node)
  {
    for (auto node1 : graph.links(node))
    {
      if (node1 < range0 or node1 >= range1)
      {
        const int remote_rank
            = std::distance(
                  node_disp.begin(),
                  std::upper_bound(node_disp.begin(), node_disp.end(), node1))
              - 1;
        assert(remote_rank < size);
        halo_node_to_remotes[node].insert(remote_rank);
      }
    }
  }

  // Loop over each halo node and count number of values to be sent to
  // each rank
  std::vector<std::int32_t> count(size, 0);
  for (const auto& halo_node : halo_node_to_remotes)
  {
    for (auto rank : halo_node.second)
      count[rank] += 2;
  }

  // Compute displacement (offsets)
  std::vector<std::int32_t> disp(size + 1, 0);
  std::partial_sum(count.begin(), count.end(), std::next(disp.begin()));

  // Pack send data for exchanging node partition data
  graph::AdjacencyList<std::int64_t> send_node_partition(
      std::vector<std::int64_t>(disp.back()), disp);
  std::vector<std::int32_t> pos(size, 0);
  for (const auto& halo_node : halo_node_to_remotes)
  {
    const std::int32_t node_index = halo_node.first;
    std::for_each(
        halo_node.second.cbegin(), halo_node.second.cend(),
        [&send_node_partition, &pos, &part, node_index, range0](auto rank)
        {
          assert(rank < send_node_partition.num_nodes());
          // (0) global node index and (1) partitioning
          auto dests = send_node_partition.links(rank);
          dests[pos[rank]++] = node_index + range0;
          dests[pos[rank]++] = part[node_index];
        });
  }

  // Do halo exchange
  const std::vector<std::int64_t> recv_node_partition
      = dolfinx::MPI::all_to_all(comm, send_node_partition).array();

  // Construct a map from all currently foreign nodes to their new
  // partition number
  std::map<std::int64_t, std::int32_t> node_ownership;
  for (std::size_t p = 0; p < recv_node_partition.size(); p += 2)
    node_ownership.insert({recv_node_partition[p], recv_node_partition[p + 1]});

  // Generate map for where new boundary nodes need to be sent
  std::map<std::int32_t, std::set<std::int32_t>> local_node_to_dests;
  for (std::int32_t node0 = 0; node0 < graph.num_nodes(); ++node0)
  {
    const std::int32_t node0_rank = part[node0];
    for (auto node1 : graph.links(node0))
    {
      std::int32_t node1_rank;
      if (node1 < range0 or node1 >= range1)
      {
        const auto it = node_ownership.find(node1);
        // remote cell - should be in map
        assert(it != node_ownership.end());
        node1_rank = it->second;
      }
      else
        node1_rank = part[node1 - range0];

      if (node0_rank != node1_rank)
        local_node_to_dests[node0].insert(node1_rank);
    }
  }

  // Convert to  AdjacencyList
  std::vector<std::int32_t> dests, offsets(1, 0);
  offsets.reserve(graph.num_nodes() + 1);
  for (std::int32_t node = 0; node < graph.num_nodes(); ++node)
  {
    dests.push_back(part[node]);
    if (const auto it = local_node_to_dests.find(node);
        it != local_node_to_dests.end())
    {
      dests.insert(dests.end(), it->second.begin(), it->second.end());
    }
    offsets.push_back(dests.size());
  }

  return graph::AdjacencyList<std::int32_t>(std::move(dests),
                                            std::move(offsets));
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
  return
      [imbalance, strategy, seed](MPI_Comm comm, int nparts,
                                  const AdjacencyList<std::int64_t>& graph,
                                  std::int32_t num_ghost_nodes, bool ghosting)
  {
    LOG(INFO) << "Compute graph partition using PT-SCOTCH";
    common::Timer timer("Compute graph partition (SCOTCH)");

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
      dests = std::vector<std::int32_t>(node_partition.begin(),
                                        node_partition.end());
    }

    // Clean up SCOTCH objects
    SCOTCH_dgraphExit(&dgrafdat);
    SCOTCH_stratExit(&strat);

    return graph::AdjacencyList<std::int32_t>(std::move(dests),
                                              std::move(offsets));
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
                              std::int32_t, bool ghosting)
  {
    LOG(INFO) << "Compute graph partition using ParMETIS";
    common::Timer timer("Compute graph partition (ParMETIS)");

    if (nparts == 1 and dolfinx::MPI::size(comm) == 1)
    {
      // Nothing to be partitioned
      return build_adjacency_list<std::int32_t>(
          std::vector<std::int32_t>(graph.num_nodes(), 0), 1);
    }

    // Build adjacency list data
    const int rank = dolfinx::MPI::rank(comm);

    // Split communicator in groups (0) without and (1) with parts of
    // the graph
    std::vector<idx_t> part(graph.num_nodes());
    MPI_Comm pcomm = MPI_COMM_NULL;
    int color = graph.num_nodes() == 0 ? 0 : 1;
    MPI_Comm_split(comm, color, rank, &pcomm);

    std::vector<idx_t> node_disp;
    if (color == 1)
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

      // Options and sata for ParMETIS
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

    if (ghosting and graph.num_nodes() > 0)
    {
      graph::AdjacencyList<std::int32_t> dest
          = compute_destination_ranks(pcomm, graph, node_disp, part);
      MPI_Comm_free(&pcomm);
      return dest;
    }
    else
    {
      MPI_Comm_free(&pcomm);
      return build_adjacency_list<std::int32_t>(
          std::vector<std::int32_t>(part.begin(), part.end()), 1);
    }
  };
}
//-----------------------------------------------------------------------------
#endif

#ifdef HAS_KAHIP

//----------------------------------------------------------------------------
std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const graph::AdjacencyList<std::int64_t>&, std::int32_t,
    bool)>
graph::kahip::partitioner(int mode, int seed, double imbalance,
                          bool suppress_output)
{
  return [mode, seed, imbalance,
          suppress_output](MPI_Comm comm, int nparts,
                           const graph::AdjacencyList<std::int64_t>& graph,
                           std::int32_t, bool ghosting)
  {
    LOG(INFO) << "Compute graph partition using (parallel) KaHIP";

    // KaHIP integer type
    using T = unsigned long long;

    common::Timer timer("Compute graph partition (KaHIP)");

    // Graph does not have vertex or adjacency weights, so we use null
    // pointers as arguments
    T *vwgt(nullptr), *adjcwgt(nullptr);

    // Build adjacency list data
    common::Timer timer1("KaHIP: build adjacency data");
    const int size = dolfinx::MPI::size(comm);
    std::vector<T> node_disp(size + 1, 0);
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
      return build_adjacency_list<std::int32_t>(
          std::vector<std::int32_t>(part.begin(), part.end()), 1);
    }
  };
}
//----------------------------------------------------------------------------
#endif
