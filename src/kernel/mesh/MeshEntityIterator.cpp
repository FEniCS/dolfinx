// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-12
// Last changed: 2006-06-03

#include <dolfin/NewMesh.h>
#include <dolfin/MeshAlgorithms.h>
#include <dolfin/MeshEntity.h>
#include <dolfin/MeshEntityIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(NewMesh& mesh, uint dim)
  : entity(mesh, dim, 0), pos(0), pos_end(mesh.size(dim)), index(0)
{
  // Compute entities if empty
  if ( pos_end == 0 )
  {
    MeshAlgorithms::computeEntities(mesh, dim);
    pos_end = mesh.size(dim);
  }
}
//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(MeshEntity& entity, uint dim)
  : entity(entity.mesh(), dim, 0), pos(0)
{
  // Get connectivity
  MeshConnectivity& c = entity.mesh().data.topology(entity.dim(), dim);

  // Compute connectivity if empty
  if ( c.size() == 0 )
    MeshAlgorithms::computeConnectivity(entity.mesh(), entity.dim(), dim);

  // Get size and index map
  if ( c.size() == 0 )
  {
    pos_end = 0;
    index = 0;
  }
  else
  {
    pos_end = c.size(entity.index());
    index = c(entity.index());
  }
}
//-----------------------------------------------------------------------------
MeshEntityIterator::MeshEntityIterator(MeshEntityIterator& it, uint dim)
  : entity(it.entity.mesh(), dim, 0), pos(0)
{
  // Get entity
  MeshEntity& entity = *it;

  // Get connectivity
  MeshConnectivity& c = entity.mesh().data.topology(entity.dim(), dim);

  // Compute connectivity if empty
  if ( c.size() == 0 )
    MeshAlgorithms::computeConnectivity(entity.mesh(), entity.dim(), dim);

  // Get size and index map
  if ( c.size() == 0 )
  {
    pos_end = 0;
    index = 0;
  }
  else
  {
    pos_end = c.size(entity.index());
    index = c(entity.index());
  }
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
