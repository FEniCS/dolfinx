// Copyright (C) 2008 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-12
// Last changed: 2009-04-01

#include <algorithm>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/main/MPI.h>
#include "UFC.h"
#include "DofMap.h"
#include "DofMapBuilder.h"

using namespace dolfin;

#if defined HAS_MPI
//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dof_map, UFC& ufc, const Mesh& mesh)
{
  // Work in progress, to be based on Algorithm 5 in the paper
  // http://home.simula.no/~logg/pub/papers/submitted-Log2008a.pdf

  message("Building parallel dof map (in parallel)");

  // Check that dof map has not been built
  if (dof_map.dof_map)
    error("Local-to-global mapping has already been computed.");

  // Allocate dof map
  const uint n = dof_map.local_dimension();
  dof_map.dof_map = new int[n*mesh.numCells()];
  
  //dolfin_not_implemented();

  // Get number of this process
  const uint this_process = MPI::process_number();
  message("Building dof map on processor %d.", this_process);

  // Build stage 0: Initialize data structures
  initialize_data_structure(dof_map, mesh);

  // Build stage 1: Compute offsets
  compute_offsets();
  
  // Build stage 2: Communicate offsets
  communicate_offsets();
  
  // Build stage 3: Compute dofs that this process is resposible for
  number_dofs();
  
  // Build stage 4: Communicate mapping on shared facets
  communicate_shared();
}
//-----------------------------------------------------------------------------
void DofMapBuilder::initialize_data_structure(DofMap& dof_map, const Mesh& mesh)
{
  const uint process_number = MPI::process_number();
  // Initialize mesh entities used by the dof map
  for (uint d = 0; d <= mesh.topology().dim(); d++)
    if (dof_map.ufc_dof_map->needs_mesh_entities(d))
      mesh.init(d);

  // Build edge-global-vertex-number information
  MeshFunction<uint>* global_vertex_indices = mesh.data().mesh_function("global vertex indices");
  if (global_vertex_indices != NULL)
  {
    std::vector<std::pair<uint, uint> > edges(mesh.topology().size(1));

    std::vector<uint> edge_vertices;
    std::pair<uint, uint> edge;
    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      std::cout << "P" << process_number << ". Edge index is " << e->index() << std::endl;
      edge_vertices.clear();
      for (VertexIterator vertex(*e); !vertex.end(); ++vertex)
      {
        std::cout << "P" << process_number << ". Vertex index is " << vertex->index() << std::endl;
        std::cout << "P" << process_number << ". Global vertex index is " << global_vertex_indices->get(vertex->index()) << std::endl;
        edge_vertices.push_back(global_vertex_indices->get(vertex->index()));
      }
      edge.first = edge_vertices[0];
      edge.second= edge_vertices[1];
      edges[e->index()] = edge;
    }

    std::cout  << "P" << process_number << ". Total number of edges is " << edges.size() << std::endl;

    // Find out which edges to ignore (belonging to a lower ranked process)
    std::map<uint, std::vector<uint> >* mapping = mesh.data().vector_mapping("overlap");

    uint num_ignored_edges = 0;
    for (uint e = 0; e < edges.size(); ++e)
    {
      uint v0 = edges[e].first;
      uint v1 = edges[e].second;
      if (mapping->count(v0) and mapping->count(v1))
      {
        std::vector<uint> intersection(1);
        std::vector<uint>::iterator intersection_end = std::set_intersection(
            (*mapping)[v0].begin(), (*mapping)[v0].end(), 
            (*mapping)[v1].begin(), (*mapping)[v1].end(), intersection.begin());

        if (intersection_end != intersection.begin() and intersection[0] < process_number)
            ++num_ignored_edges;
      }
    }

    // Compute local offset
    std::cout  << "P" << process_number << ". Number of ignored edges is " << num_ignored_edges << std::endl;
    uint local_offset = mesh.size(1) - num_ignored_edges;
    std::cout  << "P" << process_number << ". Offset is " << local_offset << std::endl;

    uint max_global_number = *std::max_element(global_vertex_indices->values(), global_vertex_indices->values() + global_vertex_indices->size());
    uint global_num_vertices = MPI::global_maximum(max_global_number);

    std::cout  << "P" << process_number << ". Max global number is " << global_num_vertices << std::endl;


    
  /*
    std::map<uint, std::vector<uint> >::iterator map_iter;

    BoundaryMesh bmesh(mesh);

    for (CellIterator cell(mesh); !cell.end(); ++cell)
      for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
      {

      }
    
  */
  }

}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_offsets()
{

}
//-----------------------------------------------------------------------------
void DofMapBuilder::communicate_offsets()
{

}
//-----------------------------------------------------------------------------
void DofMapBuilder::number_dofs()
{

}
//-----------------------------------------------------------------------------
void DofMapBuilder::communicate_shared()
{

}
//-----------------------------------------------------------------------------

#else

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dof_map, UFC& ufc, const Mesh& mesh)
{
  // Do nothing
}

//-----------------------------------------------------------------------------
void DofMapBuilder::initialize_data_structure(DofMap& dof_map, const Mesh& mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_offsets()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DofMapBuilder::communicate_offsets()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DofMapBuilder::number_dofs()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DofMapBuilder::communicate_shared()
{
  // Do nothing
}
#endif
