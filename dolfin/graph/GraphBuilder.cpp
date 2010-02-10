// Copyright (C) 2007-2008 Magnus Vikstrom and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-17
// Last changed:

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

#include <dolfin/log/log.h>
#include <dolfin/common/types.h>
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "Graph.h"
#include "GraphBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void GraphBuilder::build(Graph& graph, LocalMeshData& mesh_data)
{
  // FIXME: This is a work in progress
  warning("GraphBuilder is highly experimental.");

  cout << "Number of global cells    " << mesh_data.num_global_cells << endl;
  cout << "Number of global vertices " << mesh_data.num_global_vertices << endl;

  std::vector<std::vector<uint> >& cell_vertices = mesh_data.cell_vertices;
  const std::vector<uint>& global_cell_indices   = mesh_data.global_cell_indices;

  const uint num_gobal_cells    = mesh_data.num_global_cells;
  const uint num_local_cells    = global_cell_indices.size();
  const uint topological_dim    = mesh_data.tdim;
  const uint num_cell_facets    = topological_dim + 1;
  const uint num_facet_vertices = topological_dim;
  const uint num_cell_vertices  = topological_dim + 1;

  // Sort cell vertex indices
  std::vector<std::vector<uint> >::iterator vertices;
  for (vertices = cell_vertices.begin(); vertices != cell_vertices.end(); ++vertices)
    std::sort(vertices->begin(), vertices->end());

  // FIXME: Use std::set or std::vector?
  // Create graph (cell-cell connections are graph edges) 
  std::vector<std::set<uint> > graph_edges(num_local_cells);

  // FIXME: Pass compute_connectivity(cell_vertices0, cell_vertices1, . . .)?
  // Compute edges (cell-cell connections)
  cout << "Compute local connenctivity" << endl;
  compute_connectivity(cell_vertices, num_cell_facets, num_facet_vertices, graph_edges);

  ///-----------------------------------------------
  /// The rest only applies when running in parallel
  ///-----------------------------------------------

  cout << "Determine candidate ghost cells" << endl;
  // Determine candidate ghost cells (ghost graph vertices)
  std::vector<uint> internal_boundary_candidate;
  for (uint i = 0; i < num_local_cells; ++i)
  {
    assert(i < graph_edges.size());
    if (graph_edges[i].size() != num_cell_facets)
      internal_boundary_candidate.push_back(i);
  }  
  cout << "Number of possible boundary cells " << internal_boundary_candidate.size() << endl; 

  // Get number of possible ghost cells coming from each process
  std::vector<uint> cells_per_process(MPI::num_processes());
  cells_per_process[MPI::process_number()] =  internal_boundary_candidate.size();
  MPI::gather(cells_per_process);
  
  // Pack data for candidate ghost cells (global cell index and vertices)
  std::vector<uint> connected_cell_data;
  for (uint i = 0; i < internal_boundary_candidate.size(); ++i)
  {
    // Local cell index
    connected_cell_data.push_back(internal_boundary_candidate[i]);

    // Cell vertices
    std::vector<uint>& vertices = cell_vertices[internal_boundary_candidate[i]];
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

  // Set ghost cells on this process to zero (not communicated to self)
  cells_per_process[MPI::process_number()] = 0;

  // Distribute data to all processes
  MPI::distribute(transmit_data, partition);

  cout << "Unpack data" << endl;
  // Unpack data
  uint _offset = 0;
  std::vector<std::vector<std::vector<uint> > > off_process_cell_vertices(MPI::num_processes());
  std::vector<std::vector<uint> > off_process_local_cell_indices(MPI::num_processes());
  for (uint i = 0; i < MPI::num_processes()-1; ++i)
  {
    const uint p = partition[_offset]; 
    const uint data_length = (num_cell_vertices+1)*cells_per_process[p];

    std::vector<uint>& _local_cell_indices = off_process_local_cell_indices[p];
    std::vector<std::vector<uint> >& _cell_vertices = off_process_cell_vertices[p];

    // Loop over data for each cell
    for (uint j = _offset; j < _offset + data_length; j += num_cell_vertices+1)
    {
      assert(partition[j] == p);

      // Get cell global index
      _local_cell_indices.push_back(transmit_data[j]);

      // Get cell vertices
      std::vector<uint> vertices;
      for (uint k = 0; k < num_cell_vertices; ++k)
        vertices.push_back(transmit_data[(j+1)+k]);
      _cell_vertices.push_back(vertices);       
    }
    
    // Update offset
    _offset += data_length;
  }

  cout << "Add ghost points" << endl;
  // Add off-process (ghost) edges (cell-cell) connections to graph
  std::set<uint> ghost_cells;
  uint ghost_offset = num_local_cells;
  uint num_ghost_vertices = 0;
  for (uint i = 0; i < off_process_cell_vertices.size(); ++i)
  {
    // Local index of potential ghost cells
    const std::vector<uint>& candidate_ghost_indices = off_process_local_cell_indices[i]; 

    // Verticies of potential ghost cells on process i
    const std::vector<std::vector<uint> >& ghost_cell_vertices = off_process_cell_vertices[i]; 

    num_ghost_vertices += compute_connectivity(cell_vertices, ghost_cell_vertices, 
                                         candidate_ghost_indices, ghost_offset, 
                                         num_cell_facets, num_facet_vertices, 
                                         graph_edges, ghost_cells);
    ghost_offset += num_ghost_vertices;
  }

  cout << "Create SCOTCH data" << endl;
  compute_scotch_data(graph_edges, ghost_cells, num_gobal_cells);
  cout << "End create SCOTCH data" << endl;
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                        uint num_cell_facets,
                                        uint num_facet_vertices,
                                        std::vector<std::set<uint> >& graph_edges)
{
  std::vector<uint>::iterator it;

  // FIXME: Make this a function to avoid code duplication
  // Forward step
  for (uint i = 0; i < cell_vertices.size() - 1; ++i)
  {
    for (uint j =  cell_vertices.size()-1; j > i; --j)
    {
      if(i==j)
        error("Ooops, i==j.");

      // Find numer of vertices shared by cells i and j
      std::vector<uint> intersection(num_cell_facets);
      it = std::set_intersection(cell_vertices[i].begin(), cell_vertices[i].end(), 
                                 cell_vertices[j].begin(), cell_vertices[j].end(), 
                                 intersection.begin());
      const uint num_shared_vertices = it - intersection.begin();  

      // Insert edge if cells are neighbours
      if ( num_shared_vertices == num_facet_vertices )
        graph_edges[i].insert(j);
      else if ( num_shared_vertices > num_facet_vertices)
        error("Too many shared vertices. Cannot construct dual graph.");
    }
  }
  // Reverse step (tranpose)
  for (uint i = cell_vertices.size() - 1; i > 0; --i)
  {
    for (uint j = 0; j < i; ++j)
    {
      if(i==j)
        error("Ooops, i==j. (2)");

      // Find numer of vertices shared by cell0 and cell1
      std::vector<uint> intersection(num_cell_facets);
      it = std::set_intersection(cell_vertices[i].begin(), cell_vertices[i].end(), 
                                 cell_vertices[j].begin(), cell_vertices[j].end(), 
                                 intersection.begin());
      const uint num_shared_vertices = it - intersection.begin();  

      // Insert edge if cells are neighbours
      if ( num_shared_vertices == num_facet_vertices )
        graph_edges[i].insert(j);
      else if ( num_shared_vertices > num_facet_vertices)
        error("Too many shared vertices. Cannot construct dual graph.");
    }
  }  
}
//-----------------------------------------------------------------------------
dolfin::uint GraphBuilder::compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                        const std::vector<std::vector<uint> >& candidate_ghost_vertices,
                                        const std::vector<uint>& candidate_ghost_local_indices,
                                        const uint ghost_offset,
                                        uint num_cell_facets,
                                        uint num_facet_vertices,
                                        std::vector<std::set<uint> >& graph_edges,
                                        std::set<uint>& ghost_cells)
{
  // FIXME: This function can be made more efficient. For example, loop over local
  //        candidate cells (not all cells) and ghost candidates only. 

  std::vector<uint>::iterator it;

  // Build list of ghost vertices
  std::set<uint> ghost_vertices;
  for (uint i = 0; i < cell_vertices.size(); ++i)
  {
    for (uint j = 0; j < candidate_ghost_vertices.size(); ++j)
    {
      // Find numer of vertices shared by cells i and j
      std::vector<uint> intersection(num_cell_facets);
      it = std::set_intersection(cell_vertices[i].begin(), cell_vertices[i].end(), 
                                 candidate_ghost_vertices[j].begin(), candidate_ghost_vertices[j].end(), 
                                 intersection.begin());
      const uint num_shared_vertices = it - intersection.begin();  

      if ( num_shared_vertices == num_facet_vertices )
        ghost_vertices.insert(candidate_ghost_local_indices[j]);
      else if ( num_shared_vertices > num_facet_vertices)
        error("Too many shared vertices. Cannot construct dual graph.");
    }
  }

  // Re-number ghost vertices
  std::map<uint, uint> ghost_map;
  std::set<uint>::const_iterator gvertex;
  uint i = 0;
  for (gvertex = ghost_vertices.begin(); gvertex != ghost_vertices.end(); ++gvertex)
  {
    ghost_map[ *gvertex ] = i  + ghost_offset; 
    ++i;
  }

  // Add ghost vertices to local graph
  for (uint i = 0; i < cell_vertices.size(); ++i)
  {
    for (uint j = 0; j < candidate_ghost_vertices.size(); ++j)
    {
      // Find numer of vertices shared by cells i and j
      std::vector<uint> intersection(num_cell_facets);
      it = std::set_intersection(cell_vertices[i].begin(), cell_vertices[i].end(), 
                                 candidate_ghost_vertices[j].begin(), candidate_ghost_vertices[j].end(), 
                                 intersection.begin());
      const uint num_shared_vertices = it - intersection.begin();  

      // Insert edge if cells are neighbours
      if ( num_shared_vertices == num_facet_vertices )
        graph_edges[i].insert( ghost_map[ candidate_ghost_local_indices[j] ] );
    }
  }
  return ghost_vertices.size();
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_scotch_data(const std::vector<std::set<uint> >& graph_edges,
                               const std::set<uint>& ghost_cells,
                               uint num_global_vertices)
{
  // Number of local edges + edges connecting to ghost vertices
  uint edgelocnbr = 0;  
  std::vector<std::set<uint> >::const_iterator vertex;
  for(vertex = graph_edges.begin(); vertex != graph_edges.end(); ++vertex);
    edgelocnbr += vertex->size();

  // C-style array indexing
  //const uint baseval = 0;

  // Global data ---------------------------------

  // Total  (global) number of vertices (cells) in the graph
  //const uint vertglbnbr = num_global_vertices;

  // Total (global) number of edges (cell-cell connections) in the graph
  // Compute number of global edges
  std::vector<uint> num_global_edges(MPI::num_processes());
  num_global_edges[MPI::process_number()] = edgelocnbr;
  MPI::gather(num_global_edges);
  const uint edgeglbnbr = std::accumulate(num_global_edges.begin(), num_global_edges.end(), 0);
  cout << "Number of global edges: " << edgeglbnbr << endl;

  // Number of processes
  const uint procglbnbr = MPI::num_processes();
  cout << "Number of processes: " << procglbnbr << endl;

  // Array containing the number of local vertices on each process
  std::vector<uint> proccnttab(MPI::num_processes()); 
  proccnttab[MPI::process_number()] = graph_edges.size();
  MPI::gather(proccnttab);
  for (uint i = 0; i < MPI::process_number(); ++i)
    cout << "Testing proccnttab " << proccnttab[i] << endl;

  // Array containing . . . . 
  std::vector<uint> num_cells(MPI::num_processes());
  num_cells[MPI::process_number()] = graph_edges.size();
  MPI::gather(num_cells);
  std::vector<uint> procvrttab(MPI::num_processes() + 1);
  for (uint i = 0; i < MPI::num_processes(); ++i)
    procvrttab[i] = std::accumulate(num_cells.begin(), num_cells.begin() + i, 0);
  procvrttab[MPI::num_processes()] = procvrttab[MPI::num_processes()-1] + proccnttab[MPI::num_processes()-1];
  // Perform check
  for (uint i = 1; i <= MPI::process_number(); ++i)
  {
    cout << "Testing procvrttab: " << procvrttab[i] << "  " << procvrttab[i-1] << "  " << proccnttab[i-1] << endl;
    assert( procvrttab[i] >= (procvrttab[i-1] + proccnttab[i-1]) );
  }

  // Local data ---------------------------------

  // Number of local graph vertices (cells)
  const uint vertlocnbr = graph_edges.size();
  cout << "Number of local verticies: " << vertlocnbr << endl;

  // Number of local + ghots graph vertices (cells)
  const uint vertgstnbr = graph_edges.size() + ghost_cells.size();
  cout << "Number of edges: " << vertgstnbr << endl;

  // Still need to figure these out
  //std::vector<uint> vertloctab;
  //std::vector<uint> edgeloctab
  //std::vector<uint> edgegsttab
}
//-----------------------------------------------------------------------------
