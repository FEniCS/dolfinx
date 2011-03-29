// Copyright (C) 20011 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-03-29
// Last changed:

#include <dolfin/mesh/Mesh.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include "GenericFunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFunctionSpace::GenericFunctionSpace(boost::shared_ptr<const Mesh> mesh,
                             boost::shared_ptr<const FiniteElement> element,
                             boost::shared_ptr<const GenericDofMap> dofmap)
  : _mesh(mesh), _element(element), _dofmap(dofmap)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFunctionSpace::GenericFunctionSpace(boost::shared_ptr<const Mesh> mesh)
  : _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------

