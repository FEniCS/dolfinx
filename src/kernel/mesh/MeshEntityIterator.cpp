// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-12
// Last changed: 2006-05-12

#include <dolfin/Mesh.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(NewMesh& mesh, uint dim)
  : entity(mesh, dim, 0), pos(0), pos_end(0)
{
  // FIXME: Check special case iteration over entities that don't exist
  
  // Save end position
  //pos_end = mesh.data.topology.size(dim);
}
//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(const MeshEntityIterator& it)
  : entity(it.entity), pos(it.pos), pos_end(it.pos_end)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(MeshEntity& entity, uint dim)
  : entity(entity._mesh, dim, 0), pos(0), pos_end(0)
{
  // Set current position to start position
  //pos = mesh.data.topology.first[entity.pos];

  // FIXME: Check special case iteration over entities that don't exist

  // FIXME: Initialize connections if necessary
  
  // FIXME: Check order of + and >=

  // Save end position
}
//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(MeshEntityIterator& it, uint dim)
  : entity(it.entity._mesh, dim, 0), pos(0), pos_end(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshEntityIterator::~MeshEntityIterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
