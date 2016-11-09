// Copyright (C) 2013-2016 Anders Logg
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
// First added:  2013-09-25
// Last changed: 2016-03-02

#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/MultiMeshDofMap.h>
#include "Function.h"
#include "FunctionSpace.h"
#include "MultiMeshFunctionSpace.h"
#include "MultiMeshFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshFunction::MultiMeshFunction(std::shared_ptr<const MultiMeshFunctionSpace> V)
  : _function_space(V)
{
  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
MultiMeshFunction::MultiMeshFunction(std::shared_ptr<const MultiMeshFunctionSpace> V,
                   std::shared_ptr<GenericVector> x)
  : _function_space(V), _vector(x)
{
  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  dolfin_assert(x);
  dolfin_assert(V);
  dolfin_assert(V->dofmap());
  dolfin_assert(V->dofmap()->global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
MultiMeshFunction::~MultiMeshFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Function> MultiMeshFunction::part(std::size_t i) const
{
  // Developer note: This function has a similar role as operator[] of
  // the regular Function class.

  // Return function part if it exists in the cache
  auto it = _function_parts.find(i);
  if (it != _function_parts.end())
    return it->second;

  // Get view of function space for part
  std::shared_ptr<const FunctionSpace> V = _function_space->view(i);

  // Insert into cache and return reference
  std::shared_ptr<const Function> ui(new Function(V, _vector));
  _function_parts[i] = ui;

  return _function_parts.find(i)->second;
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> MultiMeshFunction::vector()
{
  dolfin_assert(_vector);
  return _vector;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericVector> MultiMeshFunction::vector() const
{
  dolfin_assert(_vector);
  return _vector;
}
//-----------------------------------------------------------------------------
void MultiMeshFunction::init_vector()
{
  // Developer note: At this point, this function reproduces the code
  // for the corresponding function in the Function class, but does
  // not handle distributed vectors (since we do not yet handle
  // communication between distributed bounding box trees).

  // FIXME: Dear Developer, this needs to be rewritten as in
  //        Function::init_vector()! We need to get rid of
  //        GenericVector::init(MPI_COMM_WORLD, range, local_to_global, ghost_indices);

  // Get global size
  const std::size_t N = _function_space->dofmap()->global_dimension();

  // Get local range
  const std::pair<std::size_t, std::size_t> range
    = _function_space->dofmap()->ownership_range();
  const std::size_t local_size = range.second - range.first;

  // Determine ghost vertices if dof map is distributed
  std::vector<la_index> ghost_indices;
  if (N > local_size)
    compute_ghost_indices(range, ghost_indices);

  // Create vector of dofs
  if (!_vector)
  {
    DefaultFactory factory;
    _vector = factory.create_vector(MPI_COMM_WORLD);
  }
  dolfin_assert(_vector);

  // Initialize vector of dofs
  if (_vector->empty())
  {
    //_vector->init(_function_space->mesh()->mpi_comm(), range, ghost_indices);
    std::vector<std::size_t> local_to_global(local_size);
    for (std::size_t i = 0; i < local_size; ++i)
      local_to_global[i] = i;
    _vector->init(MPI_COMM_WORLD, range, local_to_global, ghost_indices);
  }
  else
  {
    dolfin_error("Function.cpp",
                 "initialize vector of degrees of freedom for function",
                 "Cannot re-initialize a non-empty vector. Consider creating a new function");

  }

  // Set vector to zero
  _vector->zero();
}
//-----------------------------------------------------------------------------
void MultiMeshFunction::compute_ghost_indices(std::pair<std::size_t, std::size_t> range,
                                          std::vector<la_index>& ghost_indices) const
{
  // NOTE: Well, don't implement me! Rather rewrite init_vector().
  //       See Function::init_vector().
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
