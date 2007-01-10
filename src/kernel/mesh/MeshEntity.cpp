// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-11
// Last changed: 2006-10-20

#include <dolfin/dolfin_log.h>
#include <dolfin/MeshEntity.h>

using namespace dolfin;

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
    dolfin_error("Unable to compute index of given entity defined on a different mesh.");

  // Get list of entities for given topological dimension
  const uint* entities = _mesh.topology()(_dim, entity._dim)(_index);
  const uint num_entities = _mesh.topology()(_dim, entity._dim).size(_index);
  
  // Check if any entity matches
  for (uint i = 0; i < num_entities; ++i)
    if ( entities[i] == entity._index )
      return i;

  // Entity was not found
  dolfin_error("Unable to compute index of given entity (not found).");

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream,
				       const MeshEntity& entity)
{
  stream << "[ Mesh entity " << entity.index()
	 << " of topological dimension " << entity.dim() << " ]";
  return stream;
}
//-----------------------------------------------------------------------------
