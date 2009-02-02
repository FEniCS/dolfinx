// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2007, 2008.
// Modified by Martin Alnes, 2008.
//
// First added:  2008-06-18
// Last changed: 2008-12-04

#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/FunctionSpace.h>
#include "DofMap.h"
#include "Form.h"
#include "BoundaryCondition.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(const FunctionSpace& V)
  : V(reference_to_no_delete_pointer(V))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::~BoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::LocalData::LocalData(const FunctionSpace& V)
  : n(0), w(0), cell_dofs(0), facet_dofs(0)
{
  // Create array for coefficients
  n = V.dofmap().local_dimension();
  w = new double[n];
  for (uint i = 0; i < n; i++)
    w[i] = 0.0;

  // Create array for cell dofs
  cell_dofs = new uint[n];
  for (uint i = 0; i < n; i++)
    cell_dofs[i] = 0;

  // Create array for facet dofs
  const uint m = V.dofmap().num_facet_dofs();
  facet_dofs = new uint[m];
  for (uint i = 0; i < m; i++)
    facet_dofs[i] = 0;

  // Create local coordinate data
  coordinates = new double*[n];
  for (uint i = 0; i < n; i++)
  {
    coordinates[i] = new double[V.mesh().geometry().dim()];
    for (uint j = 0; j < V.mesh().geometry().dim(); j++)
      coordinates[i][j] = 0.0;
  }
}
//-----------------------------------------------------------------------------
BoundaryCondition::LocalData::~LocalData()
{
  for (uint i = 0; i < n; i++)
    delete [] coordinates[i];
  delete [] coordinates;

  delete [] w;
  delete [] cell_dofs;
  delete [] facet_dofs;
}
//-----------------------------------------------------------------------------
