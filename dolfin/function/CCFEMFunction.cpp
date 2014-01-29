// Copyright (C) 2013 Anders Logg
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
// Last changed: 2013-10-22

#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/CCFEMDofMap.h>
#include "Function.h"
#include "CCFEMFunctionSpace.h"
#include "CCFEMFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CCFEMFunction::CCFEMFunction(const CCFEMFunctionSpace& V)
  : _function_space(reference_to_no_delete_pointer(V))
{
  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
CCFEMFunction::CCFEMFunction(boost::shared_ptr<const CCFEMFunctionSpace> V)
  : _function_space(V)
{
  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
CCFEMFunction::~CCFEMFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Function& CCFEMFunction::part(std::size_t i) const
{
  // FIXME

  /*
  // Check if sub-Function is in the cache, otherwise create and add to cache
  boost::ptr_map<std::size_t, Function>::iterator sub_function
    = sub_functions.find(i);
  if (sub_function != sub_functions.end())
    return *(sub_function->second);
  else
  {
    // Extract function subspace
    std::vector<std::size_t> component = boost::assign::list_of(i);
    boost::shared_ptr<const FunctionSpace>
      sub_space(_function_space->extract_sub_space(component));

    // Insert sub-Function into map and return reference
    sub_functions.insert(i, new Function(sub_space, _vector));
    return *(sub_functions.find(i)->second);
  }
  */

  return *(_function_parts.find(0)->second);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericVector> CCFEMFunction::vector()
{
  dolfin_assert(_vector);
  return _vector;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericVector> CCFEMFunction::vector() const
{
  dolfin_assert(_vector);
  return _vector;
}
//-----------------------------------------------------------------------------
void CCFEMFunction::init_vector()
{
  // Get global size
  //const std::size_t N = _function_space->dofmap()->global_dimension();

  // Get local range
  const std::pair<std::size_t, std::size_t> range
    = _function_space->dofmap()->ownership_range();
  //const std::size_t local_size = range.second - range.first;

  // Determine ghost vertices if dof map is distributed
  std::vector<la_index> ghost_indices;
  // FIXME: Does not work in parallel
  //if (N > local_size)
  //  compute_ghost_indices(range, ghost_indices);

  // Create vector of dofs
  if (!_vector)
  {
    DefaultFactory factory;
    _vector = factory.create_vector();
  }
  dolfin_assert(_vector);

  // Initialize vector of dofs
  _vector->init(MPI_COMM_WORLD, range, ghost_indices);
  _vector->zero();
}
//-----------------------------------------------------------------------------
