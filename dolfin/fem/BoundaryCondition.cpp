// Copyright (C) 2007-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2007, 2008.
// Modified by Martin Alnes, 2008.
// Modified by Johan Hake, 2009.
//
// First added:  2008-06-18
// Last changed: 2009-11-09

#include <dolfin/common/Array.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/function/FunctionSpace.h>
#include "Form.h"
#include "BoundaryCondition.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(const FunctionSpace& V)
  : _function_space(reference_to_no_delete_pointer(V))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(boost::shared_ptr<const FunctionSpace> V)
  : _function_space(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::~BoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const FunctionSpace& BoundaryCondition::function_space() const
{
  assert(_function_space);
  return *_function_space;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const FunctionSpace> BoundaryCondition::function_space_ptr() const
{
  return _function_space;
}
//-----------------------------------------------------------------------------
void BoundaryCondition::check_arguments(GenericMatrix* A, GenericVector* b,
                                        const GenericVector* x) const
{
  assert(_function_space);

  // Check matrix and vector dimensions
  if (A && x && A->size(0) != x->size())
  {
    error("Matrix dimension (%d rows) does not match vector dimension (%d) for application of boundary conditions.",
          A->size(0), x->size());
  }

  if (A && b && A->size(0) != b->size())
  {
    error("Matrix dimension (%d rows) does not match vector dimension (%d) for application of boundary conditions.",
          A->size(0), b->size());
  }

  if (x && b && x->size() != b->size())
  {
    error("Vector dimension (%d rows) does not match vector dimension (%d) for application of boundary conditions.",
          x->size(), b->size());
  }

  // Check dimension of function space
  if (A && A->size(0) < _function_space->dim())
  {
    error("Dimension of function space (%d) too large for application of boundary conditions to linear system (%d rows).",
          _function_space->dim(), A->size(0));
  }

  if (x && x->size() < _function_space->dim())
  {
    error("Dimension of function space (%d) too large for application to boundary conditions linear system (%d rows).",
          _function_space->dim(), x->size());
  }

  if (b && b->size() < _function_space->dim())
  {
    error("Dimension of function space (%d) too large for application to boundary conditions linear system (%d rows).",
          _function_space->dim(), b->size());
  }

  // FIXME: Check case A.size() > _function_space->dim() for subspaces
}
//-----------------------------------------------------------------------------
BoundaryCondition::LocalData::LocalData(const FunctionSpace& V)
  : n(V.dofmap().max_cell_dimension()),
    w(n, 0.0),
    cell_dofs(n, 0),
    facet_dofs(V.dofmap().num_facet_dofs(), 0),
    coordinates(boost::extents[n][V.mesh()->geometry().dim()])
{
  // Do nothing
}
//-----------------------------------------------------------------------------
