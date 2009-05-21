// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-11
// Last changed: 2009-05-20

#include <dolfin/log/dolfin_log.h>
#include "MeshEntity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEntity::MeshEntity(const Mesh& mesh, uint dim, uint index)
  : _mesh(mesh), _dim(dim), _index(index)
{
  // FIXME: Add this test back, check why it breaks some demos

  /*
  if (index >= mesh.num_entities(dim))
  {
    info("Hint: Did you forget to call mesh.init(%d)?", dim);
    error("Mesh entity index %d out of range [0, %d] for entity of dimension %d.",
          index, mesh.num_entities(dim), dim);
  }
  */
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
  if ( &_mesh != &entity._mesh )
    return false;

  // Get list of entities for given topological dimension
  const uint* entities = _mesh.topology()(_dim, entity._dim)(_index);
  const uint num_entities = _mesh.topology()(_dim, entity._dim).size(_index);

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
  if ( &_mesh != &entity._mesh )
    error("Unable to compute index of given entity defined on a different mesh.");

  // Get list of entities for given topological dimension
  const uint* entities = _mesh.topology()(_dim, entity._dim)(_index);
  const uint num_entities = _mesh.topology()(_dim, entity._dim).size(_index);

  // Check if any entity matches
  for (uint i = 0; i < num_entities; ++i)
    if ( entities[i] == entity._index )
      return i;

  // Entity was not found
  error("Unable to compute index of given entity (not found).");

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream,
				       const MeshEntity& entity)
{
  stream << "[Mesh entity " << entity.index()
	 << " of topological dimension " << entity.dim() << "]";
  return stream;
}
//-----------------------------------------------------------------------------
