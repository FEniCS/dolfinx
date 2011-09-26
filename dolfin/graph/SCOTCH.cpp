// Copyright (C) 2010 Garth N. Wells
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
// First added:  2010-02-10
// Last changed: 2010-12-31

#include <algorithm>
#include <map>
#include <numeric>
#include <set>

#include <dolfin/common/Set.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/LocalMeshData.h>
#include "GraphBuilder.h"
#include "SCOTCH.h"

#ifdef HAS_SCOTCH
extern "C"
{
#include <ptscotch.h>
}
#endif

using namespace dolfin;

#ifdef HAS_SCOTCH
//-----------------------------------------------------------------------------
void SCOTCH::compute_partition(std::vector<uint>& cell_partition,
                               const LocalMeshData& mesh_data)
{
  // FIXME: Use std::set or std::vector?

  // Create data structures to hold graph
  std::vector<std::set<uint> > local_graph;
  std::set<uint> ghost_vertices;
  const uint num_global_vertices = mesh_data.num_global_cells;
  const std::vector<uint>& global_cell_indices = mesh_data.global_cell_indices;

  // Compute local dual graph
  info("Compute dual graph.");
  GraphBuilder::compute_dual_graph(mesh_data, local_graph, ghost_vertices);
  info("End compute dual graph.");

  // Compute partitions
  info("Start to compute partitions using SCOTCH");
  partition(local_graph, ghost_vertices, global_cell_indices,
            num_global_vertices, cell_partition);
  info("Finished computing partitions using SCOTCH");
}
//-----------------------------------------------------------------------------
void SCOTCH::partition(const std::vector<std::set<uint> >& local_graph,
               const std::set<uint>& ghost_vertices,
               const std::vector<uint>& global_cell_indices,
               const uint num_global_vertices,
               std::vector<uint>& cell_partition)
{
  // C-style array indexing
  const int baseval = 0;

  // Local data ---------------------------------

  // Number of local graph vertices (cells)
  const int vertlocnbr = local_graph.size();

  // Data structures for graph input to SCOTCH (add 1 for case that local graph size is zero)
  std::vector<int> vertloctab;
  vertloctab.reserve(local_graph.size() + 1);
  std::vector<int> edgeloctab;

  // Build local graph input for SCOTCH
  // (number of local + ghost graph vertices (cells),
  // number of local edges + edges connecting to ghost vertices)
  int edgelocnbr = 0;
  vertloctab.push_back(0);
  std::vector<std::set<uint> >::const_iterator vertex;
  for(vertex = local_graph.begin(); vertex != local_graph.end(); ++vertex)
  {
    edgelocnbr += vertex->size();
    vertloctab.push_back(vertloctab.back() + vertex->size());
    edgeloctab.insert(edgeloctab.end(), vertex->begin(), vertex->end());
  }

  // Handle case that local graph size is zero
  if (edgeloctab.size() == 0)
    edgeloctab.resize(1);

  // Global data ---------------------------------

  // Number of local vertices (cells) on each process
  std::vector<uint> proccnttab;
  const uint local_graph_size = local_graph.size();
  MPI::all_gather(local_graph_size, proccnttab);

  // FIXME: explain this test
  // Array containing . . . . (some sanity checks)
  std::vector<uint> procvrttab(MPI::num_processes() + 1);
  for (uint i = 0; i < MPI::num_processes(); ++i)
    procvrttab[i] = std::accumulate(proccnttab.begin(), proccnttab.begin() + i, 0);
  procvrttab[MPI::num_processes()] = procvrttab[MPI::num_processes() - 1] + proccnttab[MPI::num_processes() - 1];

  // Sanity check
  for (uint i = 1; i <= MPI::process_number(); ++i)
    assert(procvrttab[i] >= (procvrttab[i - 1] + proccnttab[i - 1]));

  /*
  // Print data ---------------
  const uint vertgstnbr = local_graph.size() + ghost_vertices.size();

  // Total  (global) number of vertices (cells) in the graph
  const SCOTCH_Num vertglbnbr = num_global_vertices;

  // Total (global) number of edges (cell-cell connections) in the graph
  std::vector<uint> num_global_edges = MPI::gather(edgelocnbr);
  const SCOTCH_Num edgeglbnbr = std::accumulate(num_global_edges.begin(), num_global_edges.end(), 0);

  // Number of processes
  const SCOTCH_Num procglbnbr = MPI::num_processes();

  cout << "Num vertices      : " << vertglbnbr << endl;
  cout << "Num edges         : " << edgeglbnbr << endl;
  cout << "Num of processes  : " << procglbnbr << endl;
  cout << "Vert per processes: " << endl;
  for (uint i = 0; i < proccnttab.size(); ++i)
    cout << proccnttab[i] << " ";
  cout << endl;
  cout << "Offests           : " << endl;
  for (uint i = 0; i < procvrttab.size(); ++i)
    cout << procvrttab[i] << "  ";
  cout << endl;

  //------ Print local data
  cout << "(*) Num vertices        : " << vertlocnbr << endl;
  cout << "(*) Num vert (inc ghost): " << vertgstnbr << endl;
  cout << "(*) Num edges           : " << edgelocnbr << endl;
  cout << "(*) Vertloctab          : " << endl;

  for (uint i = 0; i < vertloctab.size(); ++i)
    cout << vertloctab[i] << " " ;
  cout << endl;
  cout << "edgeloctab           : " << endl;
  for (uint i = 0; i < edgeloctab.size(); ++i)
    cout << edgeloctab[i] << " ";
  cout << endl;
  // -----
  */

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Create SCOTCH graph and intialise
  SCOTCH_Dgraph dgrafdat;
  if (SCOTCH_dgraphInit(&dgrafdat, *comm) != 0)
    error("Error initialising SCOTCH graph.");

  // Build SCOTCH distributed graph
  info("Start SCOTCH graph building.");
  if (SCOTCH_dgraphBuild(&dgrafdat, baseval, vertlocnbr, vertlocnbr,
                              &vertloctab[0], NULL, NULL, NULL,
                              edgelocnbr, edgelocnbr,
                              &edgeloctab[0], NULL, NULL) )
  {
    error("Error buidling SCOTCH graph.");
  }
  info("End SCOTCH graph building.");

  // Check graph
  if (SCOTCH_dgraphCheck(&dgrafdat))
    error("Consistency error in SCOTCH graph.");

  // Deal with ghost vertices (write a better description)
  SCOTCH_dgraphGhst(&dgrafdat);

  // Number of partitions (set equal to number of processes)
  const int npart = MPI::num_processes();

  // Partitioning strategy
  SCOTCH_Strat strat;
  SCOTCH_stratInit(&strat);

  // Set strategy (SCOTCH uses very crytic strings for this, and they can change between versions)
  //std::string strategy = "b{sep=m{asc=b{bnd=q{strat=f},org=q{strat=f}},low=q{strat=m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}|m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}},seq=q{strat=m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}|m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}}},seq=b{job=t,map=t,poli=S,sep=m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}|m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}}}";
  //SCOTCH_stratDgraphMap (&strat, strategy.c_str());

  // Resize vector to hold cell partition indices (ugly to handle vertlocnbr = 0 case)
  int _cell_dummy = 0;
  int* _cell_partition = 0;
  cell_partition.resize(vertlocnbr);
  if (vertlocnbr > 0)
    _cell_partition = reinterpret_cast<int*>(&cell_partition[0]);
  else
    _cell_partition = &_cell_dummy;

  // Reset SCOTCH random number generator to produce deterministic partitions
  SCOTCH_randomReset();

  // Partition graph
  info("Start SCOTCH partitioning.");
  if (SCOTCH_dgraphPart(&dgrafdat, npart, &strat, _cell_partition))
    error("Error during partitioning.");
  info("End SCOTCH partitioning.");

  // Clean up SCOTCH objects
  SCOTCH_dgraphExit(&dgrafdat);
  SCOTCH_stratExit(&strat);
}
//-----------------------------------------------------------------------------
#else
//-----------------------------------------------------------------------------
void SCOTCH::compute_partition(std::vector<uint>& cell_partition,
                               const LocalMeshData& mesh_data)
{
  error("This function requires SCOTCH.");
}
//-----------------------------------------------------------------------------
void SCOTCH::partition(const std::vector<std::set<uint> >& local_graph,
                       const std::set<uint>& ghost_vertices,
                       const std::vector<uint>& global_cell_indices,
                       uint num_global_vertices,
                       std::vector<uint>& cell_partition)
{
  error("This function requires SCOTCH.");
}
//-----------------------------------------------------------------------------
#endif
