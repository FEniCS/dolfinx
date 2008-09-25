// Copyright (C) 2008 Anders Logg (and others?).
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-09-11

#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/log.h>
#include "FunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(Mesh& mesh, FiniteElement &element, DofMap& dofmap)
  : _mesh(&mesh, NoDeleter<Mesh>()),
    _element(&element, NoDeleter<FiniteElement>()),
    _dofmap(&dofmap, NoDeleter<DofMap>())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::tr1::shared_ptr<Mesh> mesh,
                             std::tr1::shared_ptr<FiniteElement> element,
                             std::tr1::shared_ptr<DofMap> dofmap)
  : _mesh(mesh), _element(element), _dofmap(dofmap)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::~FunctionSpace()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Mesh& FunctionSpace::mesh()
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
const Mesh& FunctionSpace::mesh() const
{
  dolfin_assert(_mesh);
  return *_mesh;
}
//-----------------------------------------------------------------------------
FiniteElement& FunctionSpace::element()
{
  dolfin_assert(_element);
  return *_element;
}
//-----------------------------------------------------------------------------
const FiniteElement& FunctionSpace::element() const
{
  dolfin_assert(_element);
  return *_element;
}
//-----------------------------------------------------------------------------
DofMap& FunctionSpace::dofmap()
{
  dolfin_assert(_dofmap);
  return *_dofmap;
}
//-----------------------------------------------------------------------------
const DofMap& FunctionSpace::dofmap() const
{
  dolfin_assert(_dofmap);
  return *_dofmap;
}
//-----------------------------------------------------------------------------
