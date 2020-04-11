// Copyright (C) 2010-2013 Garth N. Wells, Anders Logg and Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SCOTCH.h"
#include "AdjacencyList.h"
#include <algorithm>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
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
dolfinx::graph::SCOTCH::compute_gps(const AdjacencyList<std::int32_t>& graph,
                                    std::size_t num_passes)
{
  // Create strategy string for Gibbs-Poole-Stockmeyer ordering
  std::string strategy = "g{pass= " + std::to_string(num_passes) + "}";
  return compute_reordering(graph, strategy);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
dolfinx::graph::SCOTCH::compute_reordering(
    const AdjacencyList<std::int32_t>& graph, std::string scotch_strategy)
{
  common::Timer timer("Compute SCOTCH graph re-ordering");

  // Number of local graph vertices
  const SCOTCH_Num vertnbr = graph.num_nodes();

  // Copy graph into array with SCOTCH_Num types
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& data = graph.array();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& offsets
      = graph.offsets();
  const std::vector<SCOTCH_Num> verttab(offsets.data(),
                                        offsets.data() + offsets.rows());
  const std::vector<SCOTCH_Num> edgetab(data.data(), data.data() + data.rows());

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

  return std::pair(std::move(permutation), std::move(inverse_permutation));
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
dolfinx::graph::SCOTCH::partition(const MPI_Comm mpi_comm, const int nparts,
                                  const AdjacencyList<SCOTCH_Num>& local_graph,
                                  const std::vector<std::size_t>& node_weights,
                                  std::int32_t num_ghost_nodes, bool ghosting)
{
  LOG(INFO) << "Compute graph partition using PT-SCOTCH";
  common::Timer timer("Compute graph partition (SCOTCH)");

  // C-style array indexing
  const SCOTCH_Num baseval = 0;

  // Local data ---------------------------------

  // Number of local graph vertices (typically cells)
  const SCOTCH_Num vertlocnbr = local_graph.num_nodes();
  const std::size_t vertgstnbr = vertlocnbr + num_ghost_nodes;

  // Get graph data. vertloctab needs to be copied to match the
  // SCOTCH_Num type.
  const SCOTCH_Num* edgeloctab = local_graph.array().data();
  const std::int32_t edgeloctab_size = local_graph.array().size();
  std::vector<SCOTCH_Num> vertloctab(local_graph.offsets().data(),
                                     local_graph.offsets().data()
                                         + local_graph.offsets().rows());

  // Global data ---------------------------------

  // Create SCOTCH graph and initialise
  SCOTCH_Dgraph dgrafdat;
  if (SCOTCH_dgraphInit(&dgrafdat, mpi_comm) != 0)
    throw std::runtime_error("Error initializing SCOTCH graph");

  // FIXME: If the nodes have weights but this rank has no nodes, then
  //        SCOTCH may deadlock since vload.data() will be nullptr on
  //        this rank but not null on all other ranks.

  // Handle cell weights (if any)
  std::vector<SCOTCH_Num> vload;
  if (!node_weights.empty())
    vload.assign(node_weights.begin(), node_weights.end());

  // Build SCOTCH distributed graph. SCOTCH is not const-correct, so we throw
  // away constness and trust SCOTCH.
  common::Timer timer1("SCOTCH: call SCOTCH_dgraphBuild");
  if (SCOTCH_dgraphBuild(&dgrafdat, baseval, vertlocnbr, vertlocnbr,
                         vertloctab.data(), nullptr, vload.data(), nullptr,
                         edgeloctab_size, edgeloctab_size,
                         const_cast<SCOTCH_Num*>(edgeloctab), nullptr, nullptr))
  {
    throw std::runtime_error("Error building SCOTCH graph");
  }
  timer1.stop();

// Check graph data for consistency
#ifdef DEBUG
  if (SCOTCH_dgraphCheck(&dgrafdat))
    throw std::runtime_error("Consistency error in SCOTCH graph");
#endif

  // Initialise partitioning strategy
  SCOTCH_Strat strat;
  SCOTCH_stratInit(&strat);

  // Set SCOTCH strategy
  // SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATDEFAULT, nparts, nparts,
  // 0.05);
  SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATSPEED, nparts, nparts, 0.05);
  // SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATQUALITY, nparts, nparts,
  // 0.05);
  // SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATSCALABILITY, nparts, nparts,
  // 0.15);

  // Resize vector to hold cell partition indices with enough extra
  // space for ghost cell partition information too. When there are no
  // nodes, vertgstnbr may be zero, and at least one dummy location
  // must be created.
  std::vector<SCOTCH_Num> cell_partition(std::max((std::size_t)1, vertgstnbr),
                                         0);

  // Reset SCOTCH random number generator to produce deterministic
  // partitions
  SCOTCH_randomReset();

  // Partition graph
  common::Timer timer2("SCOTCH: call SCOTCH_dgraphPart");
  if (SCOTCH_dgraphPart(&dgrafdat, nparts, &strat, cell_partition.data()))
    throw std::runtime_error("Error during SCOTCH partitioning");
  timer2.stop();

  // Create a map of local nodes to their additional destination processes,
  // due to ghosting. If no ghosting, this will remain empty.
  std::map<std::int32_t, std::set<std::int32_t>> local_node_to_dests;
  if (ghosting)
  {
    // Exchange halo with cell_partition data for ghosts
    // FIXME: check MPI type compatibility with SCOTCH_Num. Getting this
    //        wrong will cause a SEGV
    // FIXME: is there a better way to do this?
    MPI_Datatype MPI_SCOTCH_Num;
    if (sizeof(SCOTCH_Num) == 4)
      MPI_SCOTCH_Num = MPI_INT;
    else if (sizeof(SCOTCH_Num) == 8)
      MPI_SCOTCH_Num = MPI_LONG_LONG_INT;

    // Double check size is correct
    int tsize;
    MPI_Type_size(MPI_SCOTCH_Num, &tsize);
    assert(tsize == sizeof(SCOTCH_Num));

    common::Timer timer3("SCOTCH: call SCOTCH_dgraphHalo");
    if (SCOTCH_dgraphHalo(&dgrafdat, (void*)cell_partition.data(),
                          MPI_SCOTCH_Num))
    {
      throw std::runtime_error("Error during SCOTCH halo exchange");
    }
    timer3.stop();

    // Get SCOTCH's locally indexed graph
    common::Timer timer4("Get SCOTCH graph data");
    SCOTCH_Num* edge_ghost_tab;
    SCOTCH_dgraphData(&dgrafdat, nullptr, nullptr, nullptr, nullptr, nullptr,
                      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                      nullptr, nullptr, &edge_ghost_tab, nullptr,
                      (MPI_Comm*)&mpi_comm);
    timer4.stop();

    // Iterate through SCOTCH's local compact graph to find partition
    // boundaries and save to map
    common::Timer timer5("Extract partition boundaries from SCOTCH graph");

    // Create a map of local nodes to their additional destination processes,
    // due to ghosting. If no ghosting, this can be skipped.
    for (SCOTCH_Num i = 0; i < vertlocnbr; ++i)
    {
      const std::int32_t proc_this = cell_partition[i];

      // Get all edges outward from node i
      for (SCOTCH_Num j = vertloctab[i]; j < vertloctab[i + 1]; ++j)
      {
        // Any edge which connects to a different partition will be a ghost
        const std::int32_t proc_other = cell_partition[edge_ghost_tab[j]];
        if (proc_this != proc_other)
          local_node_to_dests[i].insert(proc_other);
      }
    }

    timer5.stop();
  }

  // Convert to offset format for AdjacencyList
  std::vector<std::int32_t> dests;
  std::vector<std::int32_t> offsets = {0};
  for (SCOTCH_Num i = 0; i < vertlocnbr; ++i)
  {
    dests.push_back(cell_partition[i]);
    const auto it = local_node_to_dests.find(i);
    if (it != local_node_to_dests.end())
      dests.insert(dests.end(), it->second.begin(), it->second.end());
    offsets.push_back(dests.size());
  }

  // Clean up SCOTCH objects
  SCOTCH_dgraphExit(&dgrafdat);
  SCOTCH_stratExit(&strat);

  return graph::AdjacencyList<std::int32_t>(dests, offsets);
}
//-----------------------------------------------------------------------------
