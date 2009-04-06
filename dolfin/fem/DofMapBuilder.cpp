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
    // Iterate over all facets (MeshEntities of co-dimension 1)
    for (MeshEntityIterator e(mesh, d-1); !e.end(); ++e)
    {
      entity_vertices.clear();
      // Get all vertices for the given facet (2 for triagles, 3 for tets)
      for (VertexIterator vertex(*e); !vertex.end(); ++vertex)
        entity_vertices.push_back(global_vertex_indices->get(vertex->index()));
      entities[e->index()] = entity_vertices;
    }

    /// Find out which edges to ignore (belonging to a lower ranked process), 
    /// which edges to number, and which edges to number and send to higher ranked processes

    std::map<uint, std::vector<uint> >* mapping = mesh.data().vector_mapping("overlap");

    // Edges numbered by other process
    std::vector<uint> ignored_entities;

    // Edges numbered by this process
    std::vector<uint> owned_entities;

    // Edges numbered by this process (key) and sendt to higher ranked process (value)
    std::map<uint, uint> owned_shared;
    for (uint e = 0; e < entities.size(); ++e)
    {
      bool ignore = false;
      bool on_boundary = true;
      const uint num_entities = entities[e].size(); // Could be moved outside loop

      // All vertices must be in the overlap to be on the boundary
      for (uint i = 0; i < num_entities; ++i)
        if (mapping->count(entities[e][i]) == 0)
            on_boundary = false;
      if (on_boundary)
      {
        // Find out if we share an edge with another process
        // This means taking the intersection between all the vertices for the given mesh entity.

        // Copy first vertex overlap
        std::vector<uint> intersection = (*mapping)[entities[e][0]];
        std::vector<uint>::iterator intersection_end = intersection.end();
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
            // FIXME: We assume the intersection is only one number here. This is wrong.
            // For example, the edge intersection for tets will sometimes be larger.
            owned_shared[e] = intersection[0];
        }
      }
      if (ignore)
        ignored_entities.push_back(e);
      else
        owned_entities.push_back(e);
    }

    // Compute local offset
    const uint local_offset = mesh.size(d-1) - ignored_entities.size();

    // Communicate all offsets
    std::vector<uint> offsets(MPI::num_processes());
    std::fill(offsets.begin(), offsets.end(), 0.0);
    offsets[process_number] = local_offset;
    MPI::gather(offsets);

    // Compute offset
    uint global_offset = 0;
    for (uint i = 0; i < process_number; ++i)
      global_offset += offsets[i];

    std::cout  << "P" << process_number << ". Offset is " << global_offset << std::endl;

    // Number owned entities
    std::vector<uint> entity_numbers(mesh.size(d-1));
    for (uint i = 0; i < owned_entities.size(); ++i)
    {
      uint entity = owned_entities[i];
      entity_numbers[entity] = global_offset + i;
    }

    // Check that we got this right
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

    // Find the global number of vertices. Should we add MeshFunction iterators to DOLFIN?
    uint max_global_number = *std::max_element(global_vertex_indices->values(), global_vertex_indices->values() + global_vertex_indices->size());
    uint global_num_vertices = MPI::global_maximum(max_global_number);

    std::cout  << "P" << process_number << ". Max global number is " << global_num_vertices << std::endl;

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
