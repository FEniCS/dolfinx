// Copyright (C) 2010-2013 Garth N. Wells, Anders Logg and Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SCOTCH.h"
#include "CSRGraph.h"
#include "GraphBuilder.h"
#include <algorithm>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/CellType.h>
#include <map>
#include <numeric>
#include <set>
#include <spdlog/spdlog.h>
#include <string>

#ifdef HAS_SCOTCH
extern "C"
{
#include <ptscotch.h>
#include <stdint.h>
}
#endif

using namespace dolfin;

#ifdef HAS_SCOTCH

//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
dolfin::graph::SCOTCH::compute_gps(const Graph& graph, std::size_t num_passes)
{
  // Create strategy string for Gibbs-Poole-Stockmeyer ordering
  std::string strategy = "g{pass= " + std::to_string(num_passes) + "}";

  return compute_reordering(graph, strategy);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
dolfin::graph::SCOTCH::compute_reordering(const Graph& graph,
                                          std::string scotch_strategy)
{
  common::Timer timer("Compute SCOTCH graph re-ordering");

  // Number of local graph vertices (cells)
  const SCOTCH_Num vertnbr = graph.size();

  // Data structures for graph input to SCOTCH (add 1 for case that
  // graph size is zero)
  std::vector<SCOTCH_Num> verttab;
  verttab.reserve(vertnbr + 1);
  std::vector<SCOTCH_Num> edgetab;
  edgetab.reserve(20 * vertnbr);

  // Build local graph input for SCOTCH
  // (number of local + ghost graph vertices (cells),
  // number of local edges + edges connecting to ghost vertices)
  SCOTCH_Num edgenbr = 0;
  verttab.push_back(0);
  Graph::const_iterator vertex;
  for (vertex = graph.begin(); vertex != graph.end(); ++vertex)
  {
    edgenbr += vertex->size();
    verttab.push_back(verttab.back() + vertex->size());
    edgetab.insert(edgetab.end(), vertex->begin(), vertex->end());
  }

  // Shrink vectors to hopefully recover an unused memory
  verttab.shrink_to_fit();
  edgetab.shrink_to_fit();

  // Create SCOTCH graph
  SCOTCH_Graph scotch_graph;

  // C-style array indexing
  const SCOTCH_Num baseval = 0;

  // Create SCOTCH graph and initialise
  if (SCOTCH_graphInit(&scotch_graph) != 0)
    throw std::runtime_error("Error initializing SCOTCH graph");

  // Build SCOTCH graph
  common::Timer timer1("SCOTCH: call SCOTCH_graphBuild");
  if (SCOTCH_graphBuild(&scotch_graph, baseval, vertnbr, &verttab[0],
                        &verttab[1], NULL, NULL, edgenbr, &edgetab[0], NULL))
  {
    throw std::runtime_error("Error building SCOTCH graph");
  }
  timer1.stop();

  // Check graph data for consistency
  /*
  #ifdef DEBUG
  if (SCOTCH_graphCheck(&scotch_graph))
    throw std::runtime_error("Consistency error in SCOTCH graph");
  #endif
  */

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
                        inverse_permutation_indices.data(), NULL, NULL, NULL))
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

  return std::make_pair(std::move(permutation), std::move(inverse_permutation));
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>
dolfin::graph::SCOTCH::partition(const MPI_Comm mpi_comm,
                                 const CSRGraph<SCOTCH_Num>& local_graph,
                                 const std::vector<std::size_t>& node_weights,
                                 std::int32_t num_ghost_nodes)
{
  spdlog::info("Compute graph partition using PT-SCOTCH");
  common::Timer timer("Compute graph partition (SCOTCH)");

  // C-style array indexing
  const SCOTCH_Num baseval = 0;

  // Number of processes
  const std::size_t num_processes = MPI::size(mpi_comm);

  // This process number
  const std::size_t proc_num = MPI::rank(mpi_comm);

  // Local data ---------------------------------

  // Number of local graph vertices (cells)
  const SCOTCH_Num vertlocnbr = local_graph.size();
  const std::size_t vertgstnbr = vertlocnbr + num_ghost_nodes;

  // Get graph data
  const std::vector<SCOTCH_Num>& edgeloctab = local_graph.edges();
  const std::vector<SCOTCH_Num>& vertloctab = local_graph.nodes();

  // Global data ---------------------------------

  // Number of local vertices (cells) on each process
  std::vector<SCOTCH_Num> proccnttab(num_processes);
  const std::vector<SCOTCH_Num>& graph_distribution
      = local_graph.node_distribution();
  for (std::size_t i = 0; i < num_processes; ++i)
    proccnttab[i] = graph_distribution[i + 1] - graph_distribution[i];

#ifdef DEBUG
  // FIXME: explain this test
  // Array containing . . . . (some sanity checks)
  std::vector<std::size_t> procvrttab(num_processes + 1);
  for (std::size_t i = 0; i < num_processes; ++i)
  {
    procvrttab[i] = std::accumulate(proccnttab.begin(), proccnttab.begin() + i,
                                    (std::size_t)0);
  }
  procvrttab[num_processes]
      = procvrttab[num_processes - 1] + proccnttab[num_processes - 1];

  // Sanity check
  for (std::size_t i = 1; i <= proc_num; ++i)
    assert(procvrttab[i] >= (procvrttab[i - 1] + proccnttab[i - 1]));
#endif

  // Create SCOTCH graph and initialise
  SCOTCH_Dgraph dgrafdat;
  if (SCOTCH_dgraphInit(&dgrafdat, mpi_comm) != 0)
    throw std::runtime_error("Error initializing SCOTCH graph");

  // Handle cell weights (if any)
  std::vector<SCOTCH_Num> vload;
  if (!node_weights.empty())
    vload.assign(node_weights.begin(), node_weights.end());

  // Build SCOTCH distributed graph. SCOTCH is not const-correct, so we throw
  // away constness and trust SCOTCH.
  common::Timer timer1("SCOTCH: call SCOTCH_dgraphBuild");
  if (SCOTCH_dgraphBuild(
          &dgrafdat, baseval, vertlocnbr, vertlocnbr,
          const_cast<SCOTCH_Num*>(vertloctab.data()), NULL, vload.data(), NULL,
          edgeloctab.size(), edgeloctab.size(),
          const_cast<SCOTCH_Num*>(edgeloctab.data()), NULL, NULL))
  {
    throw std::runtime_error("Error building SCOTCH graph");
  }
  timer1.stop();

// Check graph data for consistency
#ifdef DEBUG
  if (SCOTCH_dgraphCheck(&dgrafdat))
    throw std::runtime_error("Consistency error in SCOTCH graph");
#endif

  // Number of partitions (set equal to number of processes)
  const SCOTCH_Num npart = num_processes;

  // Initialise partitioning strategy
  SCOTCH_Strat strat;
  SCOTCH_stratInit(&strat);

  // Set SCOTCH strategy
  // SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATDEFAULT, npart, npart,
  // 0.05);
  SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATSPEED, npart, npart, 0.05);
  // SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATQUALITY, npart, npart,
  // 0.05);
  // SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATSCALABILITY, npart, npart,
  // 0.15);

  // Resize vector to hold cell partition indices with enough extra
  // space for ghost cell partition information too. When there are no
  // nodes, vertgstnbr may be zero, and at least one dummy location
  // must be created.
  std::vector<SCOTCH_Num> _cell_partition(std::max((std::size_t)1, vertgstnbr),
                                          0);

  // Reset SCOTCH random number generator to produce deterministic
  // partitions
  SCOTCH_randomReset();

  // Partition graph
  common::Timer timer2("SCOTCH: call SCOTCH_dgraphPart");
  if (SCOTCH_dgraphPart(&dgrafdat, npart, &strat, _cell_partition.data()))
    throw std::runtime_error("Error during SCOTCH partitioning");
  timer2.stop();

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
  if (SCOTCH_dgraphHalo(&dgrafdat, (void*)_cell_partition.data(),
                        MPI_SCOTCH_Num))
  {
    throw std::runtime_error("Error during SCOTCH halo exchange");
  }
  timer3.stop();

  // Get SCOTCH's locally indexed graph
  common::Timer timer4("Get SCOTCH graph data");
  SCOTCH_Num* edge_ghost_tab;
  SCOTCH_dgraphData(&dgrafdat, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL, NULL, &edge_ghost_tab, NULL,
                    (MPI_Comm*)&mpi_comm);
  timer4.stop();

  // Iterate through SCOTCH's local compact graph to find partition
  // boundaries and save to map
  common::Timer timer5("Extract partition boundaries from SCOTCH graph");
  std::map<std::int64_t, std::vector<int>> ghost_procs;
  for (SCOTCH_Num i = 0; i < vertlocnbr; ++i)
  {
    const std::size_t proc_this = _cell_partition[i];
    for (SCOTCH_Num j = vertloctab[i]; j < vertloctab[i + 1]; ++j)
    {
      const std::size_t proc_other = _cell_partition[edge_ghost_tab[j]];
      if (proc_this != proc_other)
      {
        auto map_it = ghost_procs.find(i);
        if (map_it == ghost_procs.end())
        {
          std::vector<int> sharing_processes;

          // Owning process always goes in first to vector
          sharing_processes.push_back(proc_this);
          sharing_processes.push_back(proc_other);
          ghost_procs.insert(std::make_pair(i, sharing_processes));
        }
        else
        {
          // Add to vector if not already there
          auto it = std::find(map_it->second.begin(), map_it->second.end(),
                              proc_other);
          if (it == map_it->second.end())
            map_it->second.push_back(proc_other);
        }
      }
    }
  }
  timer5.stop();

  // Clean up SCOTCH objects
  SCOTCH_dgraphExit(&dgrafdat);
  SCOTCH_stratExit(&strat);

  // Only copy the local nodes partition information. Ghost process
  // data is already in the ghost_procs map

  return std::make_pair(std::vector<int>(_cell_partition.begin(),
                                         _cell_partition.begin() + vertlocnbr),
                        std::move(ghost_procs));
}
//-----------------------------------------------------------------------------
#else
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>
dolfin::graph::SCOTCH::compute_partition(
    const MPI_Comm mpi_comm,
    const Eigen::Ref<const EigenRowArrayXXi64> cell_vertices,
    const std::vector<std::size_t>& cell_weight,
    const std::int64_t num_global_vertices, const mesh::CellType& cell_type)
{
  throw std::runtime_error(
      "DOLFIN has been configured without support for SCOTCH");
  return std::pair<std::vector<int>,
                   std::map<std::int64_t, std::vector<int>>>();
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
dolfin::graph::SCOTCH::compute_gps(const Graph& graph, std::size_t num_passes)
{
  throw std::runtime_error(
      "DOLFIN has been configured without support for SCOTCH");
  return std::make_pair(std::vector<int>(), std::vector<int>());
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
dolfin::graph::SCOTCH::compute_reordering(const Graph& graph,
                                          std::vector<int>& permutation,
                                          std::vector<int>& inverse_permutation,
                                          std::string scotch_strategy)

{
  throw std::runtime_error(
      "DOLFIN has been configured without support for SCOTCH");
  return std::make_pair(std::vector<int>(), std::vector<int>());
}
//-----------------------------------------------------------------------------

#endif
