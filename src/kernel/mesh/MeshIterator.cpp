// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Mesh.h>
#include <dolfin/MeshHierarchy.h>
#include <dolfin/MeshIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshIterator::MeshIterator(const MeshHierarchy& meshes) : it(meshes.meshes)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshIterator::MeshIterator
(const MeshHierarchy& meshes, Index index) : it(meshes.meshes, index)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshIterator::~MeshIterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshIterator& MeshIterator::operator++()
{
  ++it;
  return *this;
}
//-----------------------------------------------------------------------------
MeshIterator& MeshIterator::operator--()
{
  --it;
  return *this;
}
//-----------------------------------------------------------------------------
bool MeshIterator::end()
{
  return it.end();
}
//-----------------------------------------------------------------------------
int MeshIterator::index()
{
  return it.index();
}
//-----------------------------------------------------------------------------
MeshIterator::operator MeshPointer() const
{
  return *it;
}
//-----------------------------------------------------------------------------
Mesh& MeshIterator::operator*() const
{
  return **it;
}
//-----------------------------------------------------------------------------
Mesh* MeshIterator::operator->() const
{
  return *it;
}
//-----------------------------------------------------------------------------
