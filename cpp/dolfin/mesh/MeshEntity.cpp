// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshEntity.h"
#include "Mesh.h"
#include "MeshIterator.h"
#include "MeshTopology.h"
#include "Vertex.h"
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
void MeshEntity::init(const Mesh& mesh, std::size_t dim, std::size_t index)
{
  // Store variables
  _mesh = &mesh; // Yes, we should probably use a shared pointer here...
  _dim = dim;
  _local_index = index;

  // Check index range
  if ((std::int64_t)index < _mesh->num_entities(dim))
    return;

  // Initialize mesh entities
  _mesh->init(dim);

  // Check index range again
  if ((std::int64_t)index < _mesh->num_entities(dim))
    return;

  // Illegal index range
  dolfin_error(
      "MeshEntity.cpp", "create mesh entity",
      "Mesh entity index %d out of range [0, %d] for entity of dimension %d",
      index, _mesh->num_entities(dim), dim);
}
//-----------------------------------------------------------------------------
MeshEntity::~MeshEntity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool MeshEntity::incident(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if (_mesh != entity._mesh)
    return false;

  // Get list of entities for given topological dimension
  const std::int32_t* entities
      = _mesh->topology()(_dim, entity._dim)(_local_index);
  const std::size_t num_entities
      = _mesh->topology()(_dim, entity._dim).size(_local_index);

  // Check if any entity matches
  for (std::size_t i = 0; i < num_entities; ++i)
    if (entities[i] == entity._local_index)
      return true;

  // Entity was not found
  return false;
}
//-----------------------------------------------------------------------------
std::size_t MeshEntity::index(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if (_mesh != entity._mesh)
  {
    dolfin_error("MeshEntity.cpp", "compute index of mesh entity",
                 "Mesh entity is defined on a different mesh");
  }

  // Get list of entities for given topological dimension
  const std::int32_t* entities
      = _mesh->topology()(_dim, entity._dim)(_local_index);
  const std::size_t num_entities
      = _mesh->topology()(_dim, entity._dim).size(_local_index);

  // Check if any entity matches
  for (std::size_t i = 0; i < num_entities; ++i)
    if (entities[i] == entity._local_index)
      return i;

  // Entity was not found
  dolfin_error("MeshEntity.cpp", "compute index of mesh entity",
               "Mesh entity was not found");

  return 0;
}
//-----------------------------------------------------------------------------
geometry::Point MeshEntity::midpoint() const
{
  // Special case: a vertex is its own midpoint (don't check neighbors)
  if (_dim == 0)
    return _mesh->geometry().point(_local_index);

  // Otherwise iterate over incident vertices and compute average
  std::size_t num_vertices = 0;

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  for (auto& v : EntityRange<Vertex>(*this))
  {
    x += v.point()[0];
    y += v.point()[1];
    z += v.point()[2];
    num_vertices++;
  }

  dolfin_assert(num_vertices > 0);

  x /= double(num_vertices);
  y /= double(num_vertices);
  z /= double(num_vertices);

  geometry::Point p(x, y, z);
  return p;
}
//-----------------------------------------------------------------------------
std::uint32_t MeshEntity::owner() const
{
  if (_dim != _mesh->topology().dim())
  {
    dolfin_error("MeshEntity.cpp", "get ownership of entity",
                 "Entity ownership is only defined for cells");
  }

  const std::int32_t offset = _mesh->topology().ghost_offset(_dim);
  if (_local_index < offset)
  {
    dolfin_error("MeshEntity.cpp", "get ownership of entity",
                 "Ownership of non-ghost cells is local process");
  }

  dolfin_assert((int)_mesh->topology().cell_owner().size()
                > _local_index - offset);
  return _mesh->topology().cell_owner()[_local_index - offset];
}
//-----------------------------------------------------------------------------
std::string MeshEntity::str(bool verbose) const
{
  if (verbose)
    warning("Verbose output for MeshEntityIterator not implemented.");

  std::stringstream s;
  s << "<Mesh entity " << index() << " of topological dimension " << dim()
    << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
