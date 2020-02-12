// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshEntity.h"
#include "Geometry.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshIterator.h"
#include "Topology.h"
#include <dolfinx/common/log.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
int MeshEntity::index(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if (_mesh != entity._mesh)
    throw std::runtime_error("Mesh entity is defined on a different mesh");

  // Get list of entities for given topological dimension
  auto entities
      = _mesh->topology().connectivity(_dim, entity._dim)->links(_local_index);
  const int num_entities = _mesh->topology()
                               .connectivity(_dim, entity._dim)
                               ->num_links(_local_index);

  // Check if any entity matches
  for (int i = 0; i < num_entities; ++i)
    if (entities[i] == entity._local_index)
      return i;

  // Entity was not found
  throw std::runtime_error("Mesh entity was not found");

  return -1;
}
//-----------------------------------------------------------------------------
int MeshEntity::get_vertex_local_index(const std::int32_t v_index) const
{
  auto vertices = _mesh->topology().connectivity(_dim, 0)->links(_local_index);
  const int num_entities
      = _mesh->topology().connectivity(_dim, 0)->num_links(_local_index);
  for (int v = 0; v < num_entities; ++v)
    if (vertices[v] == v_index)
      return v;
  throw std::runtime_error("Vertex was not found");
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
