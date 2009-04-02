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
    uint d = mesh.topology().dim();
    std::vector<std::vector<uint> > entities(mesh.size(d-1));

    std::vector<uint> entity_vertices;
    std::vector<uint> entity;
    for (MeshEntityIterator e(mesh, d-1); !e.end(); ++e)
    {
      std::cout << "P" << process_number << ". Entity index is " << e->index() << std::endl;
      entity_vertices.clear();
      for (VertexIterator vertex(*e); !vertex.end(); ++vertex)
      {
        std::cout << "P" << process_number << ". Vertex index is " << vertex->index() << std::endl;
        std::cout << "P" << process_number << ". Global vertex index is " << global_vertex_indices->get(vertex->index()) << std::endl;
        entity_vertices.push_back(global_vertex_indices->get(vertex->index()));
      }
      entities[e->index()] = entity_vertices;
    }

    std::cout  << "P" << process_number << ". Total number of edges is " << entities.size() << std::endl;

    // Find out which edges to ignore (belonging to a lower ranked process)
    std::map<uint, std::vector<uint> >* mapping = mesh.data().vector_mapping("overlap");

    std::vector<uint> ignored_entities;
    std::vector<uint> owned_entities;
    std::map<uint, uint> owned_shared;
    bool ignore;
    for (uint e = 0; e < entities.size(); ++e)
    {
      ignore = false;
      bool on_boundary = true;
      uint num_entities = entities[e].size();
      for (uint i = 0; i < num_entities; ++i)
        if (mapping->count(entities[e][i]) == 0)
            on_boundary = false;
      if (on_boundary)
      {
        // Find out if we share an edge with another process
        // This means taking the intersection between all the vertices for the given mesh entity...

        // First copy first overlap
        std::vector<uint> intersection = (*mapping)[entities[e][0]];
        std::vector<uint>::iterator intersection_end = intersection.end();
        std::cout << "NUM_ENTITIES = " << num_entities << std::endl;
        for (uint i = 1; i < num_entities; ++i)
        {
          std::cout << entities[e][i] << std::endl;
          uint v = entities[e][i];
           intersection_end = std::set_intersection(
            intersection.begin(), intersection_end, 
            (*mapping)[v].begin(), (*mapping)[v].end(), intersection.begin());
        }

        // Non empty intersection
        if (intersection_end != intersection.begin())
        {
          // Ignore if shared with lower ranked process
          if (intersection[0] < process_number)
            ignore = true;
          else
            // Shared edge what we will give a number and send to process intersection[0]
            owned_shared[e] = intersection[0];
        }
      }
      if (ignore)
        ignored_entities.push_back(e);
      else
        owned_entities.push_back(e);
    }

    dolfin_assert(ignored_entities.size() + owned_entities.size() == mesh.size(d-1));

    // Compute local offset
    std::cout  << "P" << process_number << ". Number of ignored edges is " << ignored_entities.size() << std::endl;
    uint local_offset = mesh.size(d-1) - ignored_entities.size();

    // Communicate all offsets
    std::vector<uint> offsets(MPI::num_processes());
    std::fill(offsets.begin(), offsets.end(), 0.0);
    offsets[process_number] = local_offset;
    MPI::gather(offsets);

    // Compute offset
    uint real_offset = 0;
    for (uint i = 0; i < process_number; ++i)
      real_offset += offsets[i];

    std::cout  << "P" << process_number << ". Offset is " << real_offset << std::endl;

    // Number owned entities
    std::vector<uint> entity_numbers(mesh.size(d-1));
    for (uint i = 0; i < owned_entities.size(); ++i)
    {
      uint entity = owned_entities[i];
      entity_numbers[entity] = real_offset + i;
    }

    for (std::map<uint, uint>::iterator iter = owned_shared.begin(); iter != owned_shared.end(); ++iter)
    {
      uint entity = (*iter).first;
      uint process = (*iter).second;
      uint global_number = entity_numbers[entity];
      uint v0 = entities[entity][0];
      uint v1 = entities[entity][1];
      std::cout << "P" << process_number << ": Send local entity " << entity << " with global number " << global_number << " consisting of vertices " << v0 << " and " << v1
                << " to process " << process << std::endl;
    }


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
