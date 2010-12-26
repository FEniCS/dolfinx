// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-10
// Last changed:

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
//#include <boost/unordered_map.hpp>
//#include <boost/unordered_set.hpp>
#include <dolfin/common/Set.h>
#include <dolfin/common/Timer.h>
#include <dolfin/main/MPI.h>
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
               uint num_global_vertices,
               std::vector<uint>& cell_partition)
{
  // C-style array indexing
  const SCOTCH_Num baseval = 0;

  // Local data ---------------------------------

  // Number of local graph vertices (cells)
  const SCOTCH_Num vertlocnbr = local_graph.size();

  // Number of local + ghost graph vertices (cells)
  // Number of local edges + edges connecting to ghost vertices
  SCOTCH_Num edgelocnbr = 0;
  std::vector<std::set<uint> >::const_iterator vertex;
  for(vertex = local_graph.begin(); vertex != local_graph.end(); ++vertex)
    edgelocnbr += vertex->size();

  // Local graph layout
  std::vector<SCOTCH_Num> vertloctab;
  std::vector<SCOTCH_Num> edgeloctab;
  vertloctab.push_back(0);
  for (uint i = 0; i < local_graph.size(); ++i)
  {
    const std::set<uint>& vertices = local_graph[i];
    vertloctab.push_back(vertloctab[i] + vertices.size());
    edgeloctab.insert( edgeloctab.end(), vertices.begin(), vertices.end() );
  }

  // Global data ---------------------------------

  // Number of local vertices (cells) on each process
  std::vector<uint> tmp = MPI::gather(local_graph.size());
  std::vector<SCOTCH_Num> proccnttab(MPI::num_processes());
  for (uint i = 0; i < MPI::num_processes(); ++i)
    proccnttab[i] = tmp[i];

 // Array containing . . . .
  std::vector<SCOTCH_Num> procvrttab(MPI::num_processes() + 1);
  for (uint i = 0; i < MPI::num_processes(); ++i)
    procvrttab[i] = std::accumulate(proccnttab.begin(), proccnttab.begin() + i, 0);
  procvrttab[MPI::num_processes()] = procvrttab[MPI::num_processes()-1] + proccnttab[MPI::num_processes()-1];

  // Sanity check
  for (uint i = 1; i <= MPI::process_number(); ++i)
    assert( procvrttab[i] >= (procvrttab[i-1] + proccnttab[i-1]) );

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

  //------ Print data
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
  // ---------------------------
  */

  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Create SCOTCH graph and intialise
  SCOTCH_Dgraph dgrafdat;
  if (SCOTCH_dgraphInit(&dgrafdat, *comm) != 0)
    error("Error initialising SCOTCH graph.");

  // Build SCOTCH Dgraph
  if (SCOTCH_dgraphBuild(&dgrafdat, baseval, vertlocnbr, vertlocnbr,
                              &vertloctab[0], NULL, NULL, NULL,
                              edgelocnbr, edgelocnbr,
                              &edgeloctab[0], NULL, NULL) )
  {
    error("Error buidling SCOTCH graph.");
  }

  // Check graph
  if (SCOTCH_dgraphCheck(&dgrafdat))
    error("Consistency error in SCOTCH graph.");

  SCOTCH_dgraphGhst(&dgrafdat);

  // Number of partitions (set equal to number of processes)
  SCOTCH_Num npart = MPI::num_processes();

  // Partitioning strategy
  SCOTCH_Strat strat;
  SCOTCH_stratInit(&strat);

  // Set strategy (SCOTCH uses very crytic strings for this)
  //std::string strategy = "b{sep=m{asc=b{bnd=q{strat=f},org=q{strat=f}},low=q{strat=m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}|m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}},seq=q{strat=m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}|m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}}},seq=b{job=t,map=t,poli=S,sep=m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}|m{type=h,vert=80,low=h{pass=10}f{bal=0.0005,move=80},asc=b{bnd=d{dif=1,rem=1,pass=40}f{bal=0.005,move=80},org=f{bal=0.005,move=80}}}}}";
  //SCOTCH_stratDgraphMap (&strat, strategy.c_str());

  // Hold partition data
  std::vector<SCOTCH_Num> partloctab(vertlocnbr);

  info("Start SCOTCH partitioning.");
  if (SCOTCH_dgraphPart(&dgrafdat, npart, &strat, &partloctab[0]))
    error("Error during partitioning.");
  info("End SCOTCH partitioning.");

  // Clean up SCOTCH objects
  SCOTCH_dgraphExit(&dgrafdat);
  SCOTCH_stratExit(&strat);

  // Copy partiton datap
  cell_partition.resize(vertlocnbr);
  for (uint i = 0; i < cell_partition.size(); ++i)
  {
    if ( cell_partition[i] < 0 || cell_partition[i] >= (uint) npart )
      error("Problem with SCOTCH partition.");
    cell_partition[i] = partloctab[i];
  }
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
