// Copyright (C) 2006-2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Andre Massing, 2009.
//
// First added:  2006-05-11
// Last changed: 2010-02-11

#include <dolfin/log/dolfin_log.h>
#include "Mesh.h"
#include "Vertex.h"
#include "MeshEntity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEntity::MeshEntity(const Mesh& mesh, uint dim, uint index)
  : _mesh(&mesh), _dim(dim), _index(index)
{
  // Check index range
  if (index < _mesh->num_entities(dim))
    return;

  // Initialize mesh entities
  mesh.init(dim);

  // Check index range again
  if (index < _mesh->num_entities(dim))
    return;

  // Illegal index range
  error("Mesh entity index %d out of range [0, %d] for entity of dimension %d.",
        index,_mesh->num_entities(dim), dim);
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
  if ( _mesh != entity._mesh )
    return false;

  // Get list of entities for given topological dimension
  const uint* entities = _mesh->topology()(_dim, entity._dim)(_index);
  const uint num_entities = _mesh->topology()(_dim, entity._dim).size(_index);

  // Check if any entity matches
  for (uint i = 0; i < num_entities; ++i)
    if ( entities[i] == entity._index )
      return true;

  // Entity was not found
  return false;
}
//-----------------------------------------------------------------------------
dolfin::uint MeshEntity::index(const MeshEntity& entity) const
{
  // Must be in the same mesh to be incident
  if ( _mesh != entity._mesh )
    error("Unable to compute index of given entity defined on a different mesh.");

  // Get list of entities for given topological dimension
  const uint* entities = _mesh->topology()(_dim, entity._dim)(_index);
  const uint num_entities = _mesh->topology()(_dim, entity._dim).size(_index);

  // Check if any entity matches
  for (uint i = 0; i < num_entities; ++i)
    if ( entities[i] == entity._index )
      return i;

  // Entity was not found
  error("Unable to compute index of given entity (not found).");

  return 0;
}
//-----------------------------------------------------------------------------
Point MeshEntity::midpoint() const
{
  // Special case: a vertex is its own midpoint (don't check neighbors)
  if (_dim == 0)
    return _mesh->geometry().point(_index);

  // Other wise iterate over incident vertices and compute average
  uint num_vertices = 0;

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  for (VertexIterator v(*this); !v.end(); ++v)
  {
    x += v->point().x();
    y += v->point().y();
    z += v->point().z();

    num_vertices++;
  }

  x /= double(num_vertices);
  y /= double(num_vertices);
  z /= double(num_vertices);

  Point p(x, y, z);
  return p;
}
//-----------------------------------------------------------------------------
#ifdef HAS_CGAL
template <typename K>
CGAL::Bbox_3 MeshEntity::bbox () const
{
  VertexIterator v(*this);
  CGAL::Bbox_3 box(v->point().bbox<K>());
  for (++v; !v.end(); ++v)
    box = box + v->point().bbox<K>();
  return box;
}
#endif
//-----------------------------------------------------------------------------
std::string MeshEntity::str(bool verbose) const
{
  if (verbose)
    warning("Verbose output for MeshEntityIterator not implemented.");

  std::stringstream s;
  s << "<Mesh entity " << index()
    << " of topological dimension " << dim() << ">";
  return s.str();
}
//-----------------------------------------------------------------------------
