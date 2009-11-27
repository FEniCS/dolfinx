// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Andre Massing, 2009.
//
// First added:  2006-05-11
// Last changed: 2009-11-16

#include <dolfin/log/dolfin_log.h>
#include "MeshEntity.h"
#include "Vertex.h"

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
