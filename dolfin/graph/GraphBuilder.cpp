// Copyright (C) 2007-2008 Magnus Vikstrom and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-17
// Last changed:

#include <algorithm>
#include <vector>

#include <dolfin/log/log.h>
#include <dolfin/common/types.h>
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
  //const std::vector<uint>& vertex_indices      = mesh_data.cell_vertices;
  const std::vector<uint>& global_cell_indices = mesh_data.global_cell_indices;

  // If cells shares > (mesh().dim - 1) vertices, then they are connected.
  const uint topological_dim = mesh_data.tdim;

  std::vector<uint>::const_iterator cell;
  for (cell = global_cell_indices.begin(); cell != global_cell_indices.end(); ++cell)
  {  
    cout << "Cell indices " << cell - global_cell_indices.begin() << "  " <<  *cell << endl;
  }
  // Build local graph

  // Sort cell vertex indices
  std::vector<std::vector<uint> >::iterator vertices;
  for (vertices = cell_vertices.begin(); vertices != cell_vertices.end(); ++vertices)
    std::sort(vertices->begin(), vertices->end());

  // Find number of neighboring cells
  std::vector<uint> intersection(topological_dim+1);
  std::vector<uint>::iterator it;
  std::vector<std::vector<uint> >::const_iterator cell0;
  std::vector<std::vector<uint> >::const_iterator cell1;
  for (cell0 = cell_vertices.begin(); cell0 != cell_vertices.end(); ++cell0)
  {
    for (cell1 = cell0 + 1; cell1 != cell_vertices.end(); ++cell1)
    {
      it = std::set_intersection(cell0->begin(), cell0->end(), 
                                 cell1->begin(), cell1->end(), 
                                 intersection.begin());
      const uint num_shared_vertices = it - intersection.begin();  
      if ( num_shared_vertices == topological_dim - 1)
      {
        cout << "Found a neighbor: " << cell0 - cell_vertices.begin() << "  " 
              << cell1 - cell_vertices.begin() << it - intersection.begin() << endl;
      }
      else if ( num_shared_vertices > topological_dim - 1)
        error("Too many shared vertices. Cannot construct dual graph.");
    }
  }  

  // Determine possible boundary 'edges'
  
  // Exchange boundary process with neighboring process to check for connections
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
