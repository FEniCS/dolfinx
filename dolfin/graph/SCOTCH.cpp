// Copyright (C) 2010-2013 Garth N. Wells, Anders Logg and Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2011
// Modified by Chris Richardson 2013
//
// First added:  2010-02-10
// Last changed: 2014-09-09

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <string>

#include <dolfin/common/Set.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/LocalMeshData.h>
#include "GraphBuilder.h"
#include "SCOTCH.h"

#ifdef HAS_SCOTCH
extern "C"
{
#include <stdint.h>
#include <ptscotch.h>
}
#endif

using namespace dolfin;

#ifdef HAS_SCOTCH

//-----------------------------------------------------------------------------
void SCOTCH::compute_partition(const MPI_Comm mpi_comm,
                               std::vector<int>& cell_partition,
                               std::map<std::int64_t, dolfin::Set<int>>& ghost_procs,
                               const LocalMeshData& mesh_data)
{
  // Create data structures to hold graph
  std::vector<std::set<std::size_t>> local_graph;
  std::set<std::size_t> ghost_vertices;

  // Compute local dual graph
  std::unique_ptr<CellType> cell_type(CellType::create(mesh_data.cell_type));
  dolfin_assert(cell_type);
  GraphBuilder::compute_dual_graph(mpi_comm, mesh_data.cell_vertices, *cell_type,
                                   mesh_data.global_cell_indices,
                                   mesh_data.num_global_vertices, local_graph,
                                   ghost_vertices);

  // Compute partitions
  partition(mpi_comm, local_graph, mesh_data.cell_weight,
            ghost_vertices, mesh_data.global_cell_indices,
            mesh_data.num_global_cells, cell_partition, ghost_procs);

}
//-----------------------------------------------------------------------------
std::vector<int> SCOTCH::compute_gps(const Graph& graph, std::size_t num_passes)
{
  // Create strategy string for Gibbs-Poole-Stockmeyer ordering
  std::string strategy = "g{pass= " + std::to_string(num_passes) + "}";

  return compute_reordering(graph, strategy);
}
//-----------------------------------------------------------------------------
std::vector<int> SCOTCH::compute_reordering(const Graph& graph,
                                            std::string scotch_strategy)
{
  std::vector<int> permutation, inverse_permutation;
  compute_reordering(graph, permutation, inverse_permutation, scotch_strategy);
  return permutation;
}
//-----------------------------------------------------------------------------
void SCOTCH::compute_reordering(const Graph& graph,
                                std::vector<int>& permutation,
                                std::vector<int>& inverse_permutation,
                                std::string scotch_strategy)
{
  Timer timer("Compute SCOTCH graph re-ordering");

  // Number of local graph vertices (cells)
  const SCOTCH_Num vertnbr = graph.size();

  // Data structures for graph input to SCOTCH (add 1 for case that
  // graph size is zero)
  std::vector<SCOTCH_Num> verttab;
  verttab.reserve(vertnbr + 1);
  std::vector<SCOTCH_Num> edgetab;
  edgetab.reserve(20*vertnbr);

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

  // Create SCOTCH graph
  SCOTCH_Graph scotch_graph;

  // C-style array indexing
  const SCOTCH_Num baseval = 0;

  // Create SCOTCH graph and initialise
  if (SCOTCH_graphInit(&scotch_graph) != 0)
  {
    dolfin_error("SCOTCH.cpp",
                 "re-order graph using SCOTCH",
                 "Error initializing SCOTCH graph");
  }

  // Build SCOTCH graph
  Timer timer1("SCOTCH: call SCOTCH_graphBuild");
  if (SCOTCH_graphBuild(&scotch_graph, baseval,
                        vertnbr, &verttab[0], &verttab[1], NULL, NULL,
                        edgenbr, &edgetab[0], NULL))
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Error building SCOTCH graph");
  }
  timer1.stop();

  // Check graph data for consistency
  /*
  #ifdef DEBUG
  if (SCOTCH_graphCheck(&scotch_graph))
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Consistency error in SCOTCH graph");
  }
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
  Timer timer2("SCOTCH: call SCOTCH_graphOrder");
  if (SCOTCH_graphOrder(&scotch_graph, &strat, permutation_indices.data(),
                        inverse_permutation_indices.data(), NULL, NULL, NULL))
  {
    dolfin_error("SCOTCH.cpp",
                 "re-order graph using SCOTCH",
                 "Error during re-ordering");
  }
  timer2.stop();

  // Clean up SCOTCH objects
  SCOTCH_graphExit(&scotch_graph);
  SCOTCH_stratExit(&strat);

  // Copy permutation vectors
  permutation.resize(vertnbr);
  inverse_permutation.resize(vertnbr);
  std::copy(permutation_indices.begin(), permutation_indices.end(),
            permutation.begin());
  std::copy(inverse_permutation_indices.begin(),
            inverse_permutation_indices.end(), inverse_permutation.begin());
}
//-----------------------------------------------------------------------------
void SCOTCH::partition(
  const MPI_Comm mpi_comm,
  const std::vector<std::set<std::size_t>>& local_graph,
  const std::vector<std::size_t>& node_weights,
  const std::set<std::size_t>& ghost_vertices,
  const std::vector<std::size_t>& global_cell_indices,
  const std::size_t num_global_vertices,
  std::vector<int>& cell_partition,
  std::map<std::int64_t, dolfin::Set<int>>& ghost_procs)
{
  Timer timer("Compute graph partition (SCOTCH)");

  // C-style array indexing
  const SCOTCH_Num baseval = 0;

  // Number of processes
  const std::size_t num_processes = MPI::size(mpi_comm);

  // This process number
  const std::size_t proc_num = MPI::rank(mpi_comm);

  // Local data ---------------------------------

  // Number of local graph vertices (cells)
  const SCOTCH_Num vertlocnbr = local_graph.size();
  const std::size_t vertgstnbr = vertlocnbr + ghost_vertices.size();

  // Data structures for graph input to SCOTCH (add 1 for case that
  // local graph size is zero)
  std::vector<SCOTCH_Num> vertloctab;
  vertloctab.reserve(local_graph.size() + 1);
  std::vector<SCOTCH_Num> edgeloctab;

  // Build local graph input for SCOTCH
  // (number of local + ghost graph vertices (cells),
  // number of local edges + edges connecting to ghost vertices)
  SCOTCH_Num edgelocnbr = 0;
  vertloctab.push_back((SCOTCH_Num) 0);
  std::vector<std::set<std::size_t>>::const_iterator vertex;
  for(vertex = local_graph.begin(); vertex != local_graph.end(); ++vertex)
  {
    edgelocnbr += vertex->size();
    vertloctab.push_back(vertloctab.back() + vertex->size());
    edgeloctab.insert(edgeloctab.end(), vertex->begin(), vertex->end());
  }

  // Handle case that local graph size is zero
  if (edgeloctab.empty())
    edgeloctab.resize(1);

  // Global data ---------------------------------

  // Number of local vertices (cells) on each process
  std::vector<SCOTCH_Num> proccnttab;
  const SCOTCH_Num local_graph_size = local_graph.size();
  MPI::all_gather(mpi_comm, local_graph_size, proccnttab);

  #ifdef DEBUG
  // FIXME: explain this test
  // Array containing . . . . (some sanity checks)
  std::vector<std::size_t> procvrttab(num_processes + 1);
  for (std::size_t i = 0; i < num_processes; ++i)
  {
    procvrttab[i] = std::accumulate(proccnttab.begin(),
                                    proccnttab.begin() + i, (std::size_t) 0);
  }
  procvrttab[num_processes] = procvrttab[num_processes - 1]
    + proccnttab[num_processes - 1];

  // Sanity check
  for (std::size_t i = 1; i <= proc_num; ++i)
    dolfin_assert(procvrttab[i] >= (procvrttab[i - 1] + proccnttab[i - 1]));
  #endif

  // Create SCOTCH graph and initialise
  SCOTCH_Dgraph dgrafdat;
  if (SCOTCH_dgraphInit(&dgrafdat, mpi_comm) != 0)
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Error initializing SCOTCH graph");
  }

  SCOTCH_Num* veloloctab;
  std::vector<SCOTCH_Num> vload;
  if (node_weights.empty())
    veloloctab = NULL;
  else
  {
    vload.resize(node_weights.size());
    std::copy(node_weights.begin(), node_weights.end(), vload.begin());
    veloloctab = vload.data();
  }

  // Build SCOTCH distributed graph
  Timer timer1("SCOTCH: call SCOTCH_dgraphBuild");
  if (SCOTCH_dgraphBuild(&dgrafdat, baseval, vertlocnbr, vertlocnbr,
                         &vertloctab[0], NULL, veloloctab, NULL,
                         edgelocnbr, edgelocnbr,
                         &edgeloctab[0], NULL, NULL) )
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Error building SCOTCH graph");
  }
  timer1.stop();

  // Check graph data for consistency
  #ifdef DEBUG
  if (SCOTCH_dgraphCheck(&dgrafdat))
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Consistency error in SCOTCH graph");
  }
  #endif

  // Number of partitions (set equal to number of processes)
  const SCOTCH_Num npart = num_processes;

  // Initialise partitioning strategy
  SCOTCH_Strat strat;
  SCOTCH_stratInit(&strat);

  // Set SCOTCH strategy
  //SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATDEFAULT, npart, npart, 0.05);
  SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATSPEED, npart, npart, 0.05);
  //SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATQUALITY, npart, npart, 0.05);
  //SCOTCH_stratDgraphMapBuild(&strat, SCOTCH_STRATSCALABILITY, npart, npart, 0.15);

  // Resize vector to hold cell partition indices with enough extra
  // space for ghost cell partition information too When there are no
  // nodes, vertgstnbr may be zero, and at least one dummy location
  // must be created.
  std::vector<SCOTCH_Num>
    _cell_partition(std::max((std::size_t) 1, vertgstnbr), 0);

  // Reset SCOTCH random number generator to produce deterministic
  // partitions
  SCOTCH_randomReset();

  // Partition graph
  Timer timer2("SCOTCH: call SCOTCH_dgraphPart");
  if (SCOTCH_dgraphPart(&dgrafdat, npart, &strat, _cell_partition.data()))
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Error during partitioning");
  }
  timer2.stop();

  /*
  // Write SCOTCH strategy to file
  if (dolfin::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    FILE* fp;
    fp = fopen("test.txt", "w");
    SCOTCH_stratSave(&strat, fp);
    fclose(fp);
  }
  */

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
  dolfin_assert(tsize == sizeof(SCOTCH_Num));

  Timer timer3("SCOTCH: call SCOTCH_dgraphHalo");
  if (SCOTCH_dgraphHalo(&dgrafdat, (void *)_cell_partition.data(),
                        MPI_SCOTCH_Num))
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Error during halo exchange");
  }
  timer3.stop();

  // Get SCOTCH's locally indexed graph
  Timer timer4("Get SCOTCH graph data");
  SCOTCH_Num* edge_ghost_tab;
  SCOTCH_dgraphData(&dgrafdat,
                    NULL, NULL, NULL, NULL, NULL, NULL,
                    NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                    &edge_ghost_tab, NULL, (MPI_Comm *)&mpi_comm);
  timer4.stop();

  // Iterate through SCOTCH's local compact graph to find partition
  // boundaries and save to map
  Timer timer5("Extract partition boundaries from SCOTCH graph");
  for(SCOTCH_Num i = 0; i < vertlocnbr; ++i)
  {
    const std::size_t proc_this =  _cell_partition[i];
    for(SCOTCH_Num j = vertloctab[i]; j < vertloctab[i + 1]; ++j)
    {
      const std::size_t proc_other = _cell_partition[edge_ghost_tab[j]];
      if(proc_this != proc_other)
      {
        auto map_it = ghost_procs.find(i);
        if (map_it == ghost_procs.end())
        {
          dolfin::Set<int> sharing_processes;

          // Owning process goes first into dolfin::Set (unordered
          // set) so will always be first.
          sharing_processes.insert(proc_this);
          sharing_processes.insert(proc_other);
          ghost_procs.insert(std::make_pair(i, sharing_processes));
        }
        else
          map_it->second.insert(proc_other);
      }
    }
  }
  timer5.stop();

  // Clean up SCOTCH objects
  SCOTCH_dgraphExit(&dgrafdat);
  SCOTCH_stratExit(&strat);

  // Only copy the local nodes partition information. Ghost process
  // data is already in the ghost_procs map
  cell_partition.resize(vertlocnbr);
  std::copy(_cell_partition.begin(), _cell_partition.begin() + vertlocnbr,
            cell_partition.begin());
}
//-----------------------------------------------------------------------------
#else
//-----------------------------------------------------------------------------
void SCOTCH::compute_partition(const MPI_Comm mpi_comm,
                               std::vector<std::size_t>& cell_partition,
                               std::map<std::int64_t, dolfin::Set<int>>& ghost_procs,
                               const LocalMeshData& mesh_data)
{
  dolfin_error("SCOTCH.cpp",
               "partition mesh using SCOTCH",
               "DOLFIN has been configured without support for SCOTCH");
}
//-----------------------------------------------------------------------------
std::vector<int> SCOTCH::compute_gps(const Graph& graph, std::size_t num_passes)
{
  dolfin_error("SCOTCH.cpp",
               "re-order graph using SCOTCH",
               "DOLFIN has been configured without support for SCOTCH");
  return std::vector<int>();
}
//-----------------------------------------------------------------------------
std::vector<int>
SCOTCH::compute_reordering(const Graph& graph, std::string scotch_strategy)
{
  dolfin_error("SCOTCH.cpp",
               "re-order graph using SCOTCH",
               "DOLFIN has been configured without support for SCOTCH");
  return std::vector<int>();
}
//-----------------------------------------------------------------------------
void SCOTCH::compute_reordering(const Graph& graph,
                                std::vector<int>& permutation,
                                std::vector<int>& inverse_permutation,
                                std::string scotch_strategy)

{
  dolfin_error("SCOTCH.cpp",
               "re-order graph using SCOTCH",
               "DOLFIN has been configured without support for SCOTCH");
}
//-----------------------------------------------------------------------------
void SCOTCH::partition(const MPI_Comm mpi_comm,
                       const std::vector<std::set<std::size_t>>& local_graph,
                       const std::vector<std::size_t>& node_weights,
                       const std::set<std::size_t>& ghost_vertices,
                       const std::vector<std::size_t>& global_cell_indices,
                       const std::size_t num_global_vertices,
                       std::vector<int>& cell_partition,
                       std::map<std::int64_t, dolfin::Set<int>>& ghost_procs)
{
  dolfin_error("SCOTCH.cpp",
               "partition mesh using SCOTCH",
               "DOLFIN has been configured without support for SCOTCH");
}
//-----------------------------------------------------------------------------

#endif
