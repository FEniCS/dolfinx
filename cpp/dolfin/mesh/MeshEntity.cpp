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
  const int num_entities
      = _mesh->topology().connectivity(_dim, entity._dim)->size(_local_index);

  // Check if any entity matches
  for (int i = 0; i < num_entities; ++i)
    if (entities[i] == entity._local_index)
      return i;

  // Entity was not found
  throw std::runtime_error("Mesh entity was not found");

  return -1;
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
