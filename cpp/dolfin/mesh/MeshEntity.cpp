// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshEntity.h"
#include "Geometry.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshIterator.h"
#include "Topology.h"
#include <dolfin/common/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
int MeshEntity::index(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if (_mesh != entity._mesh)
    throw std::runtime_error("Mesh entity is defined on a different mesh");

  // Get list of entities for given topological dimension
  const std::int32_t* entities = _mesh->topology()
                                     .connectivity(_dim, entity._dim)
                                     ->connections(_local_index);
  // Check if any entity matches
  for (int i = 0; i < num_entities(entity._dim); ++i)
    if (entities[i] == entity._local_index)
      return i;

  // Entity was not found
  throw std::runtime_error("Mesh entity was not found");

  return -1;
}
//-----------------------------------------------------------------------------
int MeshEntity::facet_permutation(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if (_mesh != entity._mesh)
    throw std::runtime_error("Mesh entity is defined on a different mesh");

  if (entity._dim == 0)
    return 0;

  if (entity._dim == 1)
  {
    const std::vector<std::int64_t>& global_indices
      = _mesh->topology().global_indices(0);
    const std::int32_t* eg_vertices = entity.entities(0);
    const int* el_vertices = get_local_vertex_indices(entity);
    if (global_indices[eg_vertices[0]] < global_indices[eg_vertices[1]])
      return el_vertices[1] < el_vertices[0];
    else
      return el_vertices[0] < el_vertices[1];
  }

  if (entity._dim == 2)
  {
    const std::vector<std::int64_t>& global_indices
      = _mesh->topology().global_indices(0);
    const std::int32_t* eg_vertices = entity.entities(0);
    const int* el_vertices = get_local_vertex_indices(entity);

    const int num_ents = entity.num_entities(0);

    std::vector<int> v_order(num_ents);

    if (num_ents == 3)
      v_order = {0, 1, 2};
    else if (num_ents == 4)
      v_order = {0, 1, 3, 2};
    else
    {
      LOG(WARNING) << "No facet permutation was found for a facet. Integrals "
                      "containing jumps may be incorrect.";
      return 0;
    }

    // Find the index of v_order that is associated with the highest local index
    int l_num_min = -1;
    for (int v = 0; v < num_ents; ++v)
      if (l_num_min == -1 || el_vertices[v_order[v]] < el_vertices[v_order[l_num_min]])
        l_num_min = v;
    // Get the next and previous indices in v_order
    const int l_pre = l_num_min == 0 ? num_ents - 1 : l_num_min - 1;
    const int l_post = l_num_min == num_ents - 1 ? 0 : l_num_min + 1;
    const bool l_rots = el_vertices[v_order[l_post]] > el_vertices[v_order[l_pre]];

    // Find the index of v_otder that is associated with the highest global index
    int g_num_min = -1;
    for (int v = 0; v < num_ents; ++v)
      if (g_num_min == -1 || global_indices[eg_vertices[v_order[v]]] < global_indices[eg_vertices[v_order[g_num_min]]])
        g_num_min = v;
    // Get the next and previous indices in v_order
    const int g_pre = g_num_min == 0 ? num_ents - 1 : g_num_min - 1;
    const int g_post = g_num_min == num_ents - 1 ? 0 : g_num_min + 1;
    const bool g_rots = global_indices[eg_vertices[v_order[g_post]]] > global_indices[eg_vertices[v_order[g_pre]]];

    const int rot = l_num_min < g_num_min ? num_ents + l_num_min - g_num_min : l_num_min - g_num_min;
    const int ref = l_rots != g_rots;
    return 2 * rot + ref;
  }
  LOG(WARNING) << "No facet permutation was found for a facet. Integrals "
                  "containing jumps may be incorrect.";
  return 0;
}
//-----------------------------------------------------------------------------
std::string MeshEntity::str(bool verbose) const
{
  if (verbose)
    LOG(WARNING) << "Verbose output for MeshEntityIterator not implemented.";

  std::stringstream s;
  s << "<Mesh entity " << index() << " of topological dimension " << dim()
    << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
