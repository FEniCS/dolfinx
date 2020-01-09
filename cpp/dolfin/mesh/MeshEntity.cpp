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
    const int* e_vertices = get_local_vertex_indices(entity);
    return e_vertices[1] < e_vertices[0];
  }

  if (entity._dim == 2)
  {
    int num_min = -1;
    const int* e_vertices = get_local_vertex_indices(entity);
    for (int v = 0; v < entity.num_entities(0); ++v)
      if (num_min == -1 || e_vertices[v] < e_vertices[num_min])
        num_min = v;
    if (entity.num_entities(0) == 3)
    { // triangle
      const int pre = num_min == 0 ? e_vertices[entity.num_entities(0) - 1]
                                   : e_vertices[num_min - 1];
      const int post = num_min == entity.num_entities(0) - 1
                           ? e_vertices[0]
                           : e_vertices[num_min + 1];
      return 2 * num_min + (post > pre);
    }
    if (entity.num_entities(0) == 4)
    { // quadrilateral
      int mult = num_min;
      int pre = 2;
      int post = 1;
      if (num_min == 1)
      {
        pre = 0;
        post = 3;
      }
      else if (num_min == 2)
      {
        pre = 3;
        post = 0;
        mult = 3;
      }
      else if (num_min == 3)
      {
        pre = 1;
        post = 2;
        mult = 2;
      }
      return 2 * mult + (e_vertices[post] > e_vertices[pre]);
    }
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
