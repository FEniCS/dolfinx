// Copyright (C) 2013-2017 Anders Logg
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
// Last changed: 2017-10-09

#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/MultiMeshDofMap.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include "Function.h"
#include "FunctionSpace.h"
#include "MultiMeshFunctionSpace.h"
#include "MultiMeshFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshFunction::MultiMeshFunction() : Variable("u", "a function")
{
  // Do nothing
}
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

  // Create and rename function for part
  std::shared_ptr<Function> ui(new Function(V, _vector));
  ui->rename(name(), label());

  // Insert into cache
  _function_parts[i] = ui;

  return _function_parts.find(i)->second;
}
//-----------------------------------------------------------------------------
void MultiMeshFunction::assign_part(std::size_t part, const Function& v)
{
  // Replace old values with new ones
  std::size_t start_idx = 0;
  for (std::size_t j = 0; j < part; ++j)
    start_idx += _function_space->part(j)->dim();

  const std::size_t N = v.vector()->size();

  std::vector<double> buffer(N);
  std::vector<la_index> indices(N);

  // Get from [0,N)
  std::iota(indices.begin(), indices.end(), 0);
  v.vector()->get_local(buffer.data(), N, indices.data());

  // set [start_idx, N+start_idx)
  std::iota(indices.begin(), indices.end(), start_idx);
  _vector->set_local(buffer.data(), N, indices.data());

  // for (dolfin::la_index i = 0; i < (v.vector()->size()); ++i)
  //     _vector->setitem(start_idx+i, v.vector()->getitem(i));
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Function> MultiMeshFunction::part(std::size_t i,
							bool deepcopy) const
{
  if (not deepcopy)
    return part(i);

  assert(i < _function_space->num_parts());

  // Create output function
  std::shared_ptr<const FunctionSpace> V = _function_space->part(i);
  std::shared_ptr<Function> ui(new Function(V));

  // Finding the relevant part of the global vector
  std::size_t start_idx = 0;
  for (std::size_t j = 0; j < i; ++j)
  {
    start_idx += _function_space->part(j)->dim();
  }

  const std::size_t N = ui->vector()->size();
  std::vector<double> buffer(N);
  std::vector<dolfin::la_index> indices(N);

  // Get [start_idx, N+start_idx)
  std::iota(indices.begin(), indices.end(), start_idx);
  _vector->get_local(buffer.data(), N, indices.data());

  // set [0, N)
  std::iota(indices.begin(), indices.end(), 0);
  ui->vector()->set_local(buffer.data(), N, indices.data());

  // // Copy values into output function
  // for (dolfin::la_index i = 0; i < (ui->vector()->size()); ++i)
  //     ui->vector()->setitem(i, _vector->getitem(start_idx+i));

  return ui;
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
    _vector->init(range, local_to_global, ghost_indices);
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
void MultiMeshFunction::restrict(double* w, const FiniteElement& element,
                                 std::size_t part,
                                 const Cell& dolfin_cell,
                                 const double* coordinate_dofs,
                                 const ufc::cell& ufc_cell) const
{
  dolfin_assert(w);
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->dofmap());
  dolfin_assert(part < _function_space->num_parts());

  // Check if we are restricting to an element of this function space
  if (_function_space->part(part)->has_element(element)
      && _function_space->part(part)->has_cell(dolfin_cell))
  {
    // Get dofmap for cell
    const GenericDofMap& dofmap = *_function_space->dofmap()->part(part);
    const auto dofs = dofmap.cell_dofs(dolfin_cell.index());

    // Note: We should have dofmap.max_element_dofs() == dofs.size() here.
    // Pick values from vector(s)
    _vector->get_local(w, dofs.size(), dofs.data());
  }
  else
  {
    // Restrict as UFC function (by calling eval)
    restrict_as_ufc_function(w, element, part, dolfin_cell,
                             coordinate_dofs, ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void MultiMeshFunction::eval(Array<double>& values,
			     const Array<double>& x,
			     std::size_t part,
			     const ufc::cell& ufc_cell) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->multimesh());
  const Mesh& mesh = *_function_space->multimesh()->part(part);

  // Check if UFC cell comes from mesh, otherwise
  // find the cell which contains the point
  dolfin_assert(ufc_cell.mesh_identifier >= 0);
  if (ufc_cell.mesh_identifier == (int) mesh.id())
  {
    const Cell cell(mesh, ufc_cell.index);
    this->part(part)->eval(values, x, cell, ufc_cell);
  }
  else
    this->part(part)->eval(values, x);
}
//-----------------------------------------------------------------------------
void MultiMeshFunction::eval(Array<double>& values,
			     const Array<double>& x) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->multimesh());
  const MultiMesh& multimesh = *_function_space->multimesh();

  // Iterate over meshes from top to bottom
  for (std::size_t j = 0; j < multimesh.num_parts(); j++)
  {
    std::size_t part = multimesh.num_parts() - 1 - j;

    // Stop if we reached the bottom part (layer) or found layer containing point
    if (part == 0 or multimesh.part(part)->bounding_box_tree()->collides_entity(Point(x)))
    {
      this->part(part)->eval(values, x);
      break;
    }
  }
}
//-----------------------------------------------------------------------------
void MultiMeshFunction::restrict_as_ufc_function(double* w,
						 const FiniteElement& element,
						 std::size_t part,
						 const Cell& dolfin_cell,
						 const double* coordinate_dofs,
						 const ufc::cell& ufc_cell) const
{
  dolfin_assert(w);

  // Evaluate dofs to get the expansion coefficients
  element.evaluate_dofs(w, *this->part(part), coordinate_dofs, ufc_cell.orientation,
                        ufc_cell);
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
