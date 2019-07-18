// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshEntity.h"
#include "Mesh.h"
#include "MeshIterator.h"
#include "Topology.h"
#include "Vertex.h"

#include <dolfin/common/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
bool MeshEntity::incident(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if (_mesh != entity._mesh)
    return false;

  // Get list of entities for given topological dimension
  const std::int32_t* entities = _mesh->topology()
                                     .connectivity(_dim, entity._dim)
                                     ->connections(_local_index);
  const std::size_t num_entities
      = _mesh->topology().connectivity(_dim, entity._dim)->size(_local_index);

  // Check if any entity matches
  for (std::size_t i = 0; i < num_entities; ++i)
    if (entities[i] == entity._local_index)
      return true;

  // Entity was not found
  return false;
}
//-----------------------------------------------------------------------------
int MeshEntity::index(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if (_mesh != entity._mesh)
  {
    throw std::runtime_error("Mesh entity is defined on a different mesh");
  }

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

  return 0;
}
//-----------------------------------------------------------------------------
Eigen::Vector3d MeshEntity::midpoint() const
{
  // Special case: a vertex is its own midpoint (don't check neighbors)
  if (_dim == 0)
    return _mesh->geometry().x(_local_index);

  // Otherwise iterate over incident vertices and compute average
  std::size_t num_vertices = 0;

  Eigen::Vector3d x;
  x.setZero();

  for (auto& v : EntityRange<Vertex>(*this))
  {
    x += v.x();
    ++num_vertices;
  }

  assert(num_vertices > 0);
  x /= double(num_vertices);

  return x;
}
//-----------------------------------------------------------------------------
std::int32_t MeshEntity::owner() const
{
  if (_dim != _mesh->topology().dim())
  {
    throw std::runtime_error("Entity ownership is only defined for cells");
  }

  const std::int32_t offset = _mesh->topology().ghost_offset(_dim);
  if (_local_index < offset)
  {
    throw std::runtime_error("Ownership of non-ghost cells is local process");
  }

  assert((int)_mesh->topology().cell_owner().size() > _local_index - offset);
  return _mesh->topology().cell_owner()[_local_index - offset];
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
