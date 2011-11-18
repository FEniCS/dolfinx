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
// Modified by Anders Logg 2011
//
// First added:  2010-02-10
// Last changed: 2011-11-14

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

  // Compute local dual graph
  info("Compute dual graph.");
  GraphBuilder::compute_dual_graph(mesh_data, local_graph, ghost_vertices);
  info("End compute dual graph.");

  // Compute partitions
  info("Start to compute partitions using SCOTCH");
  const uint num_global_vertices = mesh_data.num_global_cells;
  const std::vector<uint>& global_cell_indices = mesh_data.global_cell_indices;
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

  // Number of processes
  const uint num_processes = MPI::num_processes();

  // This process number
  const uint proc_num = MPI::process_number();

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
  std::vector<uint> procvrttab(num_processes + 1);
  for (uint i = 0; i < num_processes; ++i)
    procvrttab[i] = std::accumulate(proccnttab.begin(), proccnttab.begin() + i, 0);
  procvrttab[num_processes] = procvrttab[num_processes - 1] + proccnttab[num_processes - 1];

  // Sanity check
  for (uint i = 1; i <= proc_num; ++i)
    dolfin_assert(procvrttab[i] >= (procvrttab[i - 1] + proccnttab[i - 1]));

  // Print graph data -------------------------------------
  const bool dislay_graph_data = false;
  if (dislay_graph_data)
  {
    const uint vertgstnbr = local_graph.size() + ghost_vertices.size();

    // Total  (global) number of vertices (cells) in the graph
    const SCOTCH_Num vertglbnbr = num_global_vertices;

    // Total (global) number of edges (cell-cell connections) in the graph
    const SCOTCH_Num edgeglbnbr = MPI::sum(edgelocnbr);


    for (uint proc = 0; proc < num_processes; ++proc)
    {
      // Print data for one process at a time
      if (proc == proc_num)
      {
        // Number of processes
        const SCOTCH_Num procglbnbr = num_processes;

        cout << "--------------------------------------------------" << endl;
        cout << "Num vertices (vertglbnbr)     : " << vertglbnbr << endl;
        cout << "Num edges (edgeglbnbr)        : " << edgeglbnbr << endl;
        cout << "Num of processes (procglbnbr) : " << procglbnbr << endl;
        cout << "Vert per processes (proccnttab) : " << endl;
        for (uint i = 0; i < proccnttab.size(); ++i)
          cout << "  " << proccnttab[i];
        cout << endl;
        cout << "Offests (procvrttab): " << endl;
        for (uint i = 0; i < procvrttab.size(); ++i)
          cout << "  " << procvrttab[i];
        cout << endl;

        //------ Print local data
        cout << "(*) Num vertices (vertlocnbr)        : " << vertlocnbr << endl;
        cout << "(*) Num vert (inc ghost) (vertgstnbr): " << vertgstnbr << endl;
        cout << "(*) Num edges (edgelocnbr)           : " << edgelocnbr << endl;
        cout << "(*) Vertloctab: " << endl;
        for (uint i = 0; i < vertloctab.size(); ++i)
          cout << "  " << vertloctab[i];
        cout << endl;
        cout << "edgeloctab: " << endl;
        for (uint i = 0; i < edgeloctab.size(); ++i)
          cout << "  " << edgeloctab[i];
        cout << endl;
        cout << "--------------------------------------------------" << endl;
      }
      MPI::barrier();
    }
    MPI::barrier();
  }
  // ------------------------------------------------------

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Create SCOTCH graph and intialise
  SCOTCH_Dgraph dgrafdat;
  if (SCOTCH_dgraphInit(&dgrafdat, *comm) != 0)
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Error initializing SCOTCH graph");
  }

  // Build SCOTCH distributed graph
  info("Start SCOTCH graph building.");
  if (SCOTCH_dgraphBuild(&dgrafdat, baseval, vertlocnbr, vertlocnbr,
                              &vertloctab[0], NULL, NULL, NULL,
                              edgelocnbr, edgelocnbr,
                              &edgeloctab[0], NULL, NULL) )
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Error building SCOTCH graph");
  }
  info("End SCOTCH graph building.");

  // Check graph data for consistency
  if (SCOTCH_dgraphCheck(&dgrafdat))
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Consistency error in SCOTCH graph");
  }

  // Number of partitions (set equal to number of processes)
  const int npart = num_processes;

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
  {
    dolfin_error("SCOTCH.cpp",
                 "partition mesh using SCOTCH",
                 "Error during partitioning");
  }
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
  dolfin_error("SCOTCH.cpp",
               "partition mesh using SCOTCH",
               "DOLFIN has been configured without support for SCOTCH");
}
//-----------------------------------------------------------------------------
void SCOTCH::partition(const std::vector<std::set<uint> >& local_graph,
                       const std::set<uint>& ghost_vertices,
                       const std::vector<uint>& global_cell_indices,
                       uint num_global_vertices,
                       std::vector<uint>& cell_partition)
{
  dolfin_error("SCOTCH.cpp",
               "partition mesh using SCOTCH",
               "DOLFIN has been configured without support for SCOTCH");
}
//-----------------------------------------------------------------------------

#endif
