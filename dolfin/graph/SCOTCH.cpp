// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-10
// Last changed:

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include <dolfin/common/Set.h>
#include <dolfin/common/Timer.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/LocalMeshData.h>
#include "SCOTCH.h"

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>

#if defined HAS_SCOTCH
extern "C"
{
#include <ptscotch.h>
}
#endif

using namespace dolfin;

#if defined HAS_SCOTCH
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
  cout << "Compute dual graph" << endl;
  compute_dual_graph(mesh_data, local_graph, ghost_vertices);
  cout << "End compute dual graph" << endl;

  // Compute partitions
  info("Start to compute partitions using SCOTCH");
  partition(local_graph, ghost_vertices, global_cell_indices, 
            num_global_vertices, cell_partition);
  info("Finihsed computing partitions using SCOTCH");
}
//-----------------------------------------------------------------------------
void SCOTCH::compute_dual_graph(const LocalMeshData& mesh_data,
                                std::vector<std::set<uint> >& local_graph,
                                std::set<uint>& ghost_vertices)
{
  // List of cell vertices
  const std::vector<std::vector<uint> >& cell_vertices = mesh_data.cell_vertices;

  const uint num_local_cells    = mesh_data.global_cell_indices.size();
  const uint topological_dim    = mesh_data.tdim;
  const uint num_cell_facets    = topological_dim + 1;
  const uint num_facet_vertices = topological_dim;
  const uint num_cell_vertices  = topological_dim + 1;

  // Resize graph (cell are graph vertices, cell-cell connections are graph edges) 
  local_graph.resize(num_local_cells);

  // Get number of cells on each process
  std::vector<uint> cells_per_process = MPI::gather(num_local_cells);

  // Compute offset for going from local to (internal) global numbering
  std::vector<uint> process_offsets(MPI::num_processes());
  for (uint i = 0; i < MPI::num_processes(); ++i)
    process_offsets[i] = std::accumulate(cells_per_process.begin(), cells_per_process.begin() + i, 0); 
  const uint process_offset = process_offsets[MPI::process_number()];

  // Compute local edges (cell-cell connections) using global (internal) numbering
  cout << "Compute local cell-cell connections" << endl;
  compute_connectivity(cell_vertices, num_facet_vertices, process_offset, 
                       local_graph);
  cout << "Finished computing local cell-cell connections" << endl;

  ///-----------------------------------------------
  /// The rest only applies when running in parallel
  ///-----------------------------------------------

  // Determine candidate ghost cells (graph ghost vertices)
  info("Preparing data to to send off-process.");
  std::vector<uint> local_boundary_cells;
  for (uint i = 0; i < num_local_cells; ++i)
  {
    assert(i < local_graph.size());
    if (local_graph[i].size() != num_cell_facets)
      local_boundary_cells.push_back(i);
  }  
  cout << "Number of possible boundary cells " << local_boundary_cells.size() << endl; 

  // Get number of possible ghost cells coming from each process
  std::vector<uint> boundary_cells_per_process = MPI::gather(local_boundary_cells.size());
  
  // Pack local data for candidate ghost cells (global cell index and vertices)
  std::vector<uint> connected_cell_data;
  for (uint i = 0; i < local_boundary_cells.size(); ++i)
  {
    // Global (internal) cell index
    connected_cell_data.push_back(local_boundary_cells[i] + process_offset);

    // Candidate cell vertices
    const std::vector<uint>& vertices = cell_vertices[local_boundary_cells[i]];
    for (uint j = 0; j < num_cell_vertices; ++j)
      connected_cell_data.push_back(vertices[j]);  
  }    

  // Prepare package to send (do not send data belonging to this process)
  std::vector<uint> partition;
  std::vector<uint> transmit_data;
  for (uint i = 0; i < MPI::num_processes(); ++i)
  {
    if(i != MPI::process_number())
    {
      transmit_data.insert(transmit_data.end(), connected_cell_data.begin(), 
                           connected_cell_data.end());
      partition.insert(partition.end(), connected_cell_data.size(), i);
    }
  }

  // Set number of candidate ghost cells on this process to zero (not communicated to self)
  boundary_cells_per_process[MPI::process_number()] = 0;

  // Distribute data to all processes
  cout << "Send off-process data" << endl;
  MPI::distribute(transmit_data, partition);
  cout << "Finished sending off-process data" << endl;

  // Data structures for unpacking data
  std::vector<std::vector<std::vector<uint> > > candidate_ghost_cell_vertices(MPI::num_processes());
  std::vector<std::vector<uint> > candidate_ghost_cell_global_indices(MPI::num_processes());

  // Unpack data
  uint _offset = 0;
  for (uint i = 0; i < MPI::num_processes()-1; ++i)
  {
    const uint p = partition[_offset]; 
    const uint data_length = (num_cell_vertices+1)*boundary_cells_per_process[p];

    std::vector<uint>& _global_cell_indices         = candidate_ghost_cell_global_indices[p];
    std::vector<std::vector<uint> >& _cell_vertices = candidate_ghost_cell_vertices[p];

    // Loop over data for each cell
    for (uint j = _offset; j < _offset + data_length; j += num_cell_vertices+1)
    {
      assert(partition[j] == p);

      // Get cell global index
      _global_cell_indices.push_back(transmit_data[j]);

      // Get cell vertices
      std::vector<uint> vertices;
      for (uint k = 0; k < num_cell_vertices; ++k)
        vertices.push_back(transmit_data[(j+1)+k]);
      _cell_vertices.push_back(vertices);       
    }
    
    // Update offset
    _offset += data_length;
  }

  // Add off-process (ghost) edges (cell-cell) connections to graph
  cout << "Compute graph ghost edges" << endl;
  std::set<uint> ghost_cell_global_indices;
  for (uint i = 0; i < candidate_ghost_cell_vertices.size(); ++i)
  {
    compute_ghost_connectivity(cell_vertices, local_boundary_cells, 
                               candidate_ghost_cell_vertices[i], 
                               candidate_ghost_cell_global_indices[i], 
                               num_cell_facets, num_facet_vertices, 
                               local_graph, ghost_cell_global_indices);
  }
  cout << "Finish compute graph ghost edges" << endl;
}
//-----------------------------------------------------------------------------
void SCOTCH::compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                  uint num_facet_vertices, uint offset,
                                  std::vector<std::set<uint> >& local_graph)
{
  // FIXME: Continue to make this function more efficient
   
  // Declare iterators
  std::vector<std::vector<uint> >::const_iterator c_vertices;
  std::vector<uint>::const_iterator vertex;
  std::vector<uint>::const_iterator c_vertex;
  std::vector<uint>::const_iterator connected_cell;

  //std::map<uint, std::vector<uint> > vertex_connectivity;
  //std::pair<std::map<uint, std::vector<uint> >::iterator, bool> ret;
  std::tr1::unordered_map<uint, std::vector<uint> > vertex_connectivity;
  std::pair<std::tr1::unordered_map<uint, std::vector<uint> >::iterator, bool> ret;

  // Build (global vertex)-(local cell) connectivity
  tic();
  for (c_vertices = cell_vertices.begin(); c_vertices != cell_vertices.end(); ++c_vertices)
  {
    const uint cell_index = c_vertices - cell_vertices.begin();
    for (vertex = c_vertices->begin(); vertex != c_vertices->end(); ++vertex)
    {
      ret = vertex_connectivity.insert(std::pair<uint, std::vector<uint> >(*vertex, std::vector<uint>()) );
      ret.first->second.push_back(cell_index);
    }
  }
  double tt = toc();
  info("Time to build vertex-cell connectivity map: %g", tt);

  tic();  
  // Iterate over all cells
  for (c_vertices = cell_vertices.begin(); c_vertices != cell_vertices.end(); ++c_vertices)
  {
     const uint index0 = c_vertices - cell_vertices.begin();

    // Iterate over cell vertices
    for (c_vertex = c_vertices->begin(); c_vertex != c_vertices->end(); ++c_vertex)
    {
      // Iterate over cells connected to this vertex
      for (connected_cell = vertex_connectivity[*c_vertex].begin(); connected_cell != vertex_connectivity[*c_vertex].end(); ++connected_cell)
      {
        const uint index1 = *connected_cell;
        if (index0 == index1)
          break;
 
        // Vertices of candidate neighbour
        const std::vector<uint>& candidate_vertices = cell_vertices[*connected_cell];

        uint num_common_vertices = 0;
        for (vertex = c_vertices->begin(); vertex != c_vertices->end(); ++vertex) 
        {
          if (std::find(candidate_vertices.begin(), candidate_vertices.end(), *vertex) != candidate_vertices.end())
            ++num_common_vertices; 
          if (num_common_vertices == num_facet_vertices)
          {
            local_graph[index0].insert(index1 + offset);
            local_graph[index1].insert(index0 + offset);
            break;
          }  
        }
      }
    }   
  }
  tt = toc();
  info("Time to build local dual graph: : %g", tt);
}
//-----------------------------------------------------------------------------
dolfin::uint SCOTCH::compute_ghost_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                          const std::vector<uint>& local_boundary_cells,
                                          const std::vector<std::vector<uint> >& candidate_ghost_vertices,
                                          const std::vector<uint>& candidate_ghost_global_indices,
                                          uint num_cell_facets,
                                          uint num_facet_vertices,
                                          std::vector<std::set<uint> >& local_graph,
                                          std::set<uint>& ghost_cells)
{
  // FIXME: This function needs to be made more efficient. 

  const uint num_ghost_vertices_0 = ghost_cells.size();

  std::vector<uint>::iterator it;
  for (uint i = 0; i < local_boundary_cells.size(); ++i)
  {
    const uint local_cell_index = local_boundary_cells[i];
    for (uint j = 0; j < candidate_ghost_vertices.size(); ++j)
    {
      // Find numer of vertices shared by cells i and j
      std::vector<uint> intersection(num_cell_facets);
      it = std::set_intersection(cell_vertices[local_cell_index].begin(), cell_vertices[local_cell_index].end(), 
                                 candidate_ghost_vertices[j].begin(), candidate_ghost_vertices[j].end(), 
                                 intersection.begin());
      const uint num_shared_vertices = it - intersection.begin();  

      if ( num_shared_vertices == num_facet_vertices )
      {
        local_graph[local_cell_index].insert(candidate_ghost_global_indices[j]);
        ghost_cells.insert(candidate_ghost_global_indices[j]);
      }
      else if ( num_shared_vertices > num_facet_vertices)
        error("Too many shared vertices. Cannot construct dual graph.");
    }
  }
  
  // Return number of newly added ghost vertices
  return ghost_cells.size() - num_ghost_vertices_0;
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
  const uint vertgstnbr = local_graph.size() + ghost_vertices.size();

  // Total  (global) number of vertices (cells) in the graph
  const SCOTCH_Num vertglbnbr = num_global_vertices;

  // Total (global) number of edges (cell-cell connections) in the graph
  std::vector<uint> num_global_edges = MPI::gather(edgelocnbr);
  const SCOTCH_Num edgeglbnbr = std::accumulate(num_global_edges.begin(), num_global_edges.end(), 0);

  // Number of processes
  const SCOTCH_Num procglbnbr = MPI::num_processes();

  //------ Print global data
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
*/
  // Construct communicator (copy of MPI_COMM_WORLD)
  MPICommunicator comm;

  // Create SCOTCH graph and intialise
  SCOTCH_Dgraph dgrafdat;
  if (SCOTCH_dgraphInit(&dgrafdat, *comm) != 0)
    error("Error initialising SCOTCH graph.");

  // Build SCOTCH Dgraph
  int err= SCOTCH_dgraphBuild(&dgrafdat, baseval, vertlocnbr, vertlocnbr,
                              &vertloctab[0], NULL, NULL, NULL,
                              edgelocnbr, edgelocnbr,
                              &edgeloctab[0], NULL, NULL);
  if (err != 0)
    error("Error buidling SCOTCH graph.");

  // FIXME: Is this required?
  //err = SCOTCH_dgraphGhst(&dgrafdat);
  //if (err != 0)
  //  error("Error buidling SCOTCH ghost points.");

  // Check graph 
  if (SCOTCH_dgraphCheck(&dgrafdat))
    error("Consistency error in SCOTCH graph.");

  // Number of partitions
  SCOTCH_Num npart = MPI::num_processes();

  // Partitioning strategy
  SCOTCH_Strat strat;
  SCOTCH_stratInit(&strat);

  // Hold partition data
  std::vector<SCOTCH_Num> partloctab(vertlocnbr);

  cout << "Start SCOTCH partitioning " << endl;
  err = SCOTCH_dgraphPart(&dgrafdat, npart, &strat, &partloctab[0]);
  if (err)
    warning("Error during partitioning.");
  cout << "End SCOTCH partitioning " << endl;

  // Free SCOTCH graph
  SCOTCH_dgraphExit(&dgrafdat);

  // Copy partiton data
  cell_partition.resize(vertlocnbr);
  for (uint i = 0; i < partloctab.size(); ++i)
    cell_partition[i] = partloctab[i];
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
void SCOTCH::compute_dual_graph(const LocalMeshData& mesh_data,
                                std::vector<std::set<uint> >& local_graph,
                                std::set<uint>& ghost_vertices)
{
  error("This function requires SCOTCH.");
}
//-----------------------------------------------------------------------------
void SCOTCH::compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                  uint num_facet_vertices, uint offset,
                                  std::vector<std::set<uint> >& graph)
{
  error("This function requires SCOTCH.");
}
//-----------------------------------------------------------------------------
dolfin::uint SCOTCH::compute_ghost_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                                const std::vector<uint>& local_boundary_cells,
                                                const std::vector<std::vector<uint> >& candidate_ghost_vertices,
                                                const std::vector<uint>& candidate_ghost_global_indices,
                                                uint num_cell_facets, uint num_facet_vertices,
                                                std::vector<std::set<uint> >& ghost_graph_edges,
                                                std::set<uint>& ghost_cells)
{
  error("This function requires SCOTCH.");
  return 0;
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

