// Copyright (C) 2007-2008 Magnus Vikstrom and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-17
// Last changed:

#include <algorithm>
#include <iterator>
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
void GraphBuilder::build(Graph& graph, LocalMeshData& mesh_data, 
                         Graph::Representation rep)
{
  // FIXME: This is a work in progress

  cout << "Number of global cells    " << mesh_data.num_global_cells << endl;
  cout << "Number of global vertices " << mesh_data.num_global_vertices << endl;

  std::vector<std::vector<uint> >& cell_vertices = mesh_data.cell_vertices;
  const std::vector<uint>& global_cell_indices   = mesh_data.global_cell_indices;


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

  // FIXME: Use local or global cell numbering?
  // FIXME: Pass compute_connectivity(cell_vertices0, cell_vertices1, . . .)?
  // Compute edges (cell-cell connections)
  compute_connectivity(cell_vertices, global_cell_indices, num_cell_facets, 
                       num_facet_vertices, graph_edges);


  ///-----------------------------------------------
  /// The rest only applies when running in parallel
  ///-----------------------------------------------

  // Determine canidate ghost cells (ghost graph vertices)
  std::vector<uint> internal_boundary_candidate;
  std::vector<std::set<uint> >::const_iterator graph_vertex;
  for (graph_vertex = graph_edges.begin(); graph_vertex != graph_edges.end(); ++graph_vertex)
  {
    if (graph_vertex->size() != num_cell_facets)
      internal_boundary_candidate.push_back(graph_vertex-graph_edges.begin());
  }  
  cout << "Number of possible boundary cells " << internal_boundary_candidate.size() << endl; 

  // Get number of possible ghost cells coming from each process
  std::vector<uint> cells_per_process(MPI::num_processes());
  cells_per_process[MPI::process_number()] =  internal_boundary_candidate.size();
  MPI::gather(cells_per_process);
  
  // Pack data for possible ghost cells (global cell index and vertices)
  std::vector<uint> connected_cell_data;
  for (uint i = 0; i < internal_boundary_candidate.size(); ++i)
  {
    // Global cell index
    const uint global_cell_index = global_cell_indices[internal_boundary_candidate[i]];
    connected_cell_data.push_back(global_cell_index);

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

  // Unpack data
  uint offset = 0;
  std::vector<std::vector<std::vector<uint> > > off_process_cell_vertices(MPI::num_processes());
  std::vector<std::vector<uint> > off_process_global_cell_indices(MPI::num_processes());
  for (uint i = 0; i < MPI::num_processes()-1; ++i)
  {
    const uint p = partition[offset]; 
    const uint data_length = (num_cell_vertices+1)*cells_per_process[p];

    std::vector<uint>& _global_cell_indices = off_process_global_cell_indices[p];
    std::vector<std::vector<uint> >& _cell_vertices = off_process_cell_vertices[p];

    // Loop over data for each cell
    for (uint j = offset; j < offset + data_length; j += num_cell_vertices+1)
    {
      //cout << "Testing " << partition[j] << "  " << p << "  " << j - offset << " " << cells_per_process[i] << endl;
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
    offset += data_length;
  }

  // FIXME: There may be duplicate ghost cells???

  // Add off-process (ghost) edges (cell-cell) connections to graph
  for (uint i = 0; i < off_process_cell_vertices.size(); ++i)
  {
    const std::vector<std::vector<uint> >& ghost_cell_vertices = off_process_cell_vertices[i]; 
    const std::vector<uint>& ghost_global_cell_indices = off_process_global_cell_indices[i]; 
    compute_connectivity(cell_vertices, ghost_cell_vertices, ghost_global_cell_indices,
                         num_cell_facets, num_facet_vertices, graph_edges);
  }


  warning("GraphBuilder is highly experimental.");
}
//-----------------------------------------------------------------------------
void GraphBuilder::build(Graph& graph, const Mesh& mesh, Graph::Representation rep)
{
  // Clear graph
  graph.clear();

  // Set type
  graph._type = Graph::directed;

  // Build
  if(rep == Graph::dual)
    create_dual(graph, mesh);
  else if(rep == Graph::nodal)
    create_nodal(graph, mesh);
  else
    error("Graph type unknown");
}
//-----------------------------------------------------------------------------
void GraphBuilder::create_nodal(Graph& graph, const Mesh& mesh)
{
  error("Partitioning of nodal mesh graphs probably doesn't work. Please test and fix.");

  // Initialise mesh
  mesh.init(0, 0);

  // Get number of vertices and edges
  uint num_vertices = mesh.init(0);
  uint num_edges    = 2*mesh.init(1);

  // Initialise graph
  graph.init(num_vertices, num_edges);

  // Create nodal graph. Iterate over edges from all vertices
  uint i = 0, j = 0;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    graph._vertices[i++] = j;
    const uint* entities = vertex->entities(0);
    for (uint k = 0; k < vertex->num_entities(0); k++)
      graph._edges[j++] = entities[k];

    // Replace with this?
    /*
    for (VertexIterator neighbor(vertex); !neighbor.end(); ++neighbor)
    {
      //dolfin_debug1("Edge no %d", j);
		dolfin_debug2("Vertex no %d connected to vertex no %d", vertex->index(), neighbor->index());
      edges[j++] = neighbor->index();
    }
    */
  }
}
//-----------------------------------------------------------------------------
void GraphBuilder::create_dual(Graph& graph, const Mesh& mesh)
{
  // Initialise mesh
  uint D = mesh.topology().dim();
  mesh.init(D, D);
  const MeshConnectivity& connectivity = mesh.topology()(D,D);

  // Get number of vertices and edges
  uint num_vertices = mesh.num_cells();
  uint num_edges   = connectivity.size();

  // Initialise graph
  graph.init(num_vertices, num_edges);

  // Create dual graph. Iterate over neighbors from all cells
  uint i = 0, j = 0;
  for (CellIterator c0(mesh); !c0.end(); ++c0)
  {
    graph._vertices[i++] = j;
    for (CellIterator c1(*c0); !c1.end(); ++c1)
      graph._edges[j++] = c1->index();
  }
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                        const std::vector<uint>& global_cell_indices,
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
        graph_edges[i].insert(global_cell_indices[j]);
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
void GraphBuilder::compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                        const std::vector<std::vector<uint> >& ghost_cell_vertices,
                                        const std::vector<uint>& ghost_global_cell_indices,
                                        uint num_cell_facets,
                                        uint num_facet_vertices,
                                        std::vector<std::set<uint> >& graph_edges)
{
  std::vector<uint>::iterator it;
  for (uint i = 0; i < cell_vertices.size(); ++i)
  {
    for (uint j = 0; j < ghost_cell_vertices.size(); ++j)
    {
      // Find numer of vertices shared by cells i and j
      std::vector<uint> intersection(num_cell_facets);
      it = std::set_intersection(cell_vertices[i].begin(), cell_vertices[i].end(), 
                                 ghost_cell_vertices[j].begin(), ghost_cell_vertices[j].end(), 
                                 intersection.begin());
      const uint num_shared_vertices = it - intersection.begin();  

      // Insert edge if cells are neighbours
      if ( num_shared_vertices == num_facet_vertices )
      {
        cout << "Adding off-process edges" << endl;
        graph_edges[i].insert(ghost_global_cell_indices[j]);
      }
      else if ( num_shared_vertices > num_facet_vertices)
        error("Too many shared vertices. Cannot construct dual graph.");
    }
  }
}
//-----------------------------------------------------------------------------
