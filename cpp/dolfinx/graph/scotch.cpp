// Copyright (C) 2010-2020 Garth N. Wells, Anders Logg and Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "scotch.h"
#include "AdjacencyList.h"
#include <algorithm>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <map>
#include <numeric>
#include <set>
#include <string>

extern "C"
{
#include <ptscotch.h>
#include <stdint.h>
}

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
graph::scotch::compute_gps(const AdjacencyList<std::int32_t>& graph,
                           std::size_t num_passes)
{
  // Create strategy string for Gibbs-Poole-Stockmeyer ordering
  std::string strategy = "g{pass=" + std::to_string(num_passes) + "}";
  return compute_reordering(graph, strategy);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
graph::scotch::compute_reordering(const AdjacencyList<std::int32_t>& graph,
                                  std::string scotch_strategy)
{
  common::Timer timer("Compute SCOTCH graph re-ordering");

  // Number of local graph vertices
  const SCOTCH_Num vertnbr = graph.num_nodes();

  // Copy graph into array with SCOTCH_Num types
  const std::vector<std::int32_t>& data = graph.array();
  const std::vector<std::int32_t>& offsets = graph.offsets();
  const std::vector<SCOTCH_Num> verttab(offsets.begin(), offsets.end());
  const std::vector<SCOTCH_Num> edgetab(data.begin(), data.end());

  // Create SCOTCH graph
  SCOTCH_Graph scotch_graph;

  // C-style array indexing
  const SCOTCH_Num baseval = 0;

  // Create SCOTCH graph and initialise
  if (SCOTCH_graphInit(&scotch_graph) != 0)
    throw std::runtime_error("Error initializing SCOTCH graph");

  // Build SCOTCH graph
  SCOTCH_Num edgenbr = verttab.back();
  common::Timer timer1("SCOTCH: call SCOTCH_graphBuild");
  if (SCOTCH_graphBuild(&scotch_graph, baseval, vertnbr, verttab.data(),
                        nullptr, nullptr, nullptr, edgenbr, edgetab.data(),
                        nullptr))
  {
    throw std::runtime_error("Error building SCOTCH graph");
  }
  timer1.stop();

// Check graph data for consistency
#ifdef DEBUG
  if (SCOTCH_graphCheck(&scotch_graph))
    throw std::runtime_error("Consistency error in SCOTCH graph");
#endif

  // Re-ordering strategy
  SCOTCH_Strat strat;
  SCOTCH_stratInit(&strat);

  // Set SCOTCH strategy (if provided)
  if (!scotch_strategy.empty())
    SCOTCH_stratGraphOrder(&strat, scotch_strategy.c_str());

  // Vector to hold permutation vectors
  std::vector<SCOTCH_Num> permutation_indices(vertnbr);
  std::vector<SCOTCH_Num> inverse_permutation_indices(vertnbr);

  // Reset SCOTCH random number generator to produce deterministic
  // partitions
  SCOTCH_randomReset();

  // Compute re-ordering
  common::Timer timer2("SCOTCH: call SCOTCH_graphOrder");
  if (SCOTCH_graphOrder(&scotch_graph, &strat, permutation_indices.data(),
                        inverse_permutation_indices.data(), nullptr, nullptr,
                        nullptr))
  {
    throw std::runtime_error("Error during SCOTCH re-ordering");
  }
  timer2.stop();

  // Clean up SCOTCH objects
  SCOTCH_graphExit(&scotch_graph);
  SCOTCH_stratExit(&strat);

  // Copy permutation vectors
  std::vector<int> permutation(vertnbr);
  std::vector<int> inverse_permutation(vertnbr);
  std::copy(permutation_indices.begin(), permutation_indices.end(),
            permutation.begin());
  std::copy(inverse_permutation_indices.begin(),
            inverse_permutation_indices.end(), inverse_permutation.begin());

  return {std::move(permutation), std::move(inverse_permutation)};
}
//-----------------------------------------------------------------------------
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
#ifdef DEBUG
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
      throw("Unknown SCOTCH strategy");
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
//-----------------------------------------------------------------------------
