// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-12
// Last changed: 2006-05-31

#include <dolfin/Mesh.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(NewMesh& mesh, uint dim)
  : entity(mesh, dim, 0), pos(0), pos_end(mesh.size(dim)), index(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(const MeshEntityIterator& it)
  : entity(it.entity), pos(it.pos), pos_end(it.pos_end), index(it.index)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(MeshEntity& entity, uint dim)
  : entity(entity.mesh(), dim, 0), pos(0)
{
  // Get connectivity
  MeshConnectivity& c = entity.mesh().data.topology(entity.dim(), dim);

  // Get size and index map
  pos_end = c.size(entity.index());
  index = c(entity.index());
}
//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(MeshEntityIterator& it, uint dim)
  : entity(it.entity._mesh, dim, 0), pos(0)
{
  // Get connectivity
  MeshConnectivity& c = it.entity.mesh().data.topology(it.entity.dim(), dim);

  // Get size and index map
  pos_end = c.size(it.entity.index());
  index = c(it.entity.index());
}
//-----------------------------------------------------------------------------
MeshEntityIterator::~MeshEntityIterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream,
				       const MeshEntityIterator& it)
{
  stream << "[ Mesh entity iterator at position "
	 << it.pos
	 << " stepping from 0 to "
	 << it.pos_end - 1
	 << " ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
