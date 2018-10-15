// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Function.h"
#include "FunctionSpace.h"
#include <algorithm>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <unordered_map>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V) : _function_space(V)
{
  // Check that we don't have a subspace
  if (!V->component().empty())
  {
    throw std::runtime_error("Cannot create Function from subspace. Consider "
                             "collapsing the function space");
  }

  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V,
                   std::shared_ptr<la::PETScVector> x)
    : _function_space(V), _vector(x)
{
  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  assert(V->dofmap());
  assert(V->dofmap()->global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v)
{
  // Make a copy of all the data, or if v is a sub-function, then we
  // collapse the dof map and copy only the relevant entries from the
  // vector of v.
  assert(v._vector);
  if (v._vector->size() == v._function_space->dim())
  {
    // Copy function space pointer
    this->_function_space = v._function_space;

    // Copy vector
    this->_vector = std::make_shared<la::PETScVector>(*v._vector);
  }
  else
  {
    // Create new collapsed FunctionSpace
    std::unordered_map<std::size_t, std::size_t> collapsed_map;
    std::tie(_function_space, collapsed_map) = v._function_space->collapse();

    // Get row indices of original and new vectors
    std::unordered_map<std::size_t, std::size_t>::const_iterator entry;
    std::vector<PetscInt> new_rows(collapsed_map.size());
    std::vector<PetscInt> old_rows(collapsed_map.size());
    std::size_t i = 0;
    for (entry = collapsed_map.begin(); entry != collapsed_map.end(); ++entry)
    {
      new_rows[i] = entry->first;
      old_rows[i++] = entry->second;
    }

    // Gather values into a vector
    assert(v.vector());
    std::vector<PetscScalar> gathered_values(collapsed_map.size());
    v.vector()->get_local(gathered_values.data(), gathered_values.size(),
                          old_rows.data());

    // Initial new vector (global)
    init_vector();
    assert(_function_space->dofmap());
    assert(_vector->size() == _function_space->dofmap()->global_dimension());

    // FIXME (local): Check this for local or global
    // Set values in vector
    this->_vector->set_local(gathered_values.data(), collapsed_map.size(),
                             new_rows.data());
    this->_vector->apply();
  }
}
//-----------------------------------------------------------------------------
/*
const Function& Function::operator= (const Function& v)
{
  assert(v._vector);

  // Make a copy of all the data, or if v is a sub-function, then we
  // collapse the dof map and copy only the relevant entries from the
  // vector of v.
  if (v._vector->size() == v._function_space->dim())
  {
    // Copy function space
    _function_space = v._function_space;

    // Copy vector
    _vector = v._vector->copy();

    // Clear subfunction cache
    _sub_functions.clear();
  }
  else
  {
    // Create new collapsed FunctionSpace
    std::unordered_map<std::size_t, std::size_t> collapsed_map;
    _function_space = v._function_space->collapse(collapsed_map);

    // Get row indices of original and new vectors
    std::unordered_map<std::size_t, std::size_t>::const_iterator entry;
    std::vector<PetscInt> new_rows(collapsed_map.size());
    std::vector<PetscInt> old_rows(collapsed_map.size());
    std::size_t i = 0;
    for (entry = collapsed_map.begin(); entry != collapsed_map.end(); ++entry)
    {
      new_rows[i]   = entry->first;
      old_rows[i++] = entry->second;
    }

    // Gather values into a vector
    assert(v.vector());
    std::vector<double> gathered_values(collapsed_map.size());
    v.vector()->get_local(gathered_values.data(), gathered_values.size(),
                          old_rows.data());

    // Initial new vector (global)
    init_vector();
    assert(_function_space->dofmap());
    assert(_vector->size()
                  == _function_space->dofmap()->global_dimension());

    // FIXME (local): Check this for local or global
    // Set values in vector
    this->_vector->set_local(gathered_values.data(), collapsed_map.size(),
                             new_rows.data());
    this->_vector->apply("insert");
  }

  return *this;
}
*/
//-----------------------------------------------------------------------------
Function Function::sub(std::size_t i) const
{
  // Extract function subspace
  auto sub_space = _function_space->sub({i});

  // Return sub-function
  assert(sub_space);
  assert(_vector);
  return Function(sub_space, _vector);
}
//-----------------------------------------------------------------------------
std::shared_ptr<la::PETScVector> Function::vector()
{
  assert(_vector);
  assert(_function_space->dofmap());

  // Check that this is not a sub function.
  if (_vector->size() != _function_space->dofmap()->global_dimension())
  {
    throw std::runtime_error(
        "Cannot access a non-const vector from a subfunction");
  }

  return _vector;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const la::PETScVector> Function::vector() const
{
  assert(_vector);
  return _vector;
}
//-----------------------------------------------------------------------------
void Function::eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                        values,
                    const Eigen::Ref<const EigenRowArrayXXd> x) const
{
  assert(_function_space);
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();

  // Find the cell that contains x
  for (unsigned int i = 0; i < x.rows(); ++i)
  {
    const double* _x = x.row(i).data();
    const geometry::Point point(mesh.geometry().dim(), _x);

    // Get index of first cell containing point
    unsigned int id
        = mesh.bounding_box_tree()->compute_first_entity_collision(point, mesh);

    // If not found, use the closest cell
    if (id == std::numeric_limits<unsigned int>::max())
    {
      // Check if the closest cell is within DOLFIN_EPS. This we can
      // allow without _allow_extrapolation
      std::pair<unsigned int, double> close
          = mesh.bounding_box_tree()->compute_closest_entity(point, mesh);

      if (close.second < DOLFIN_EPS)
        id = close.first;
      else
      {
        throw std::runtime_error("Cannot evaluate function at point. The point "
                                 "is not inside the domain.");
      }
    }

    // Create cell that contains point
    const mesh::Cell cell(mesh, id);

    // Call evaluate function
    eval(values.row(i), x.row(i), cell);
  }
}
//-----------------------------------------------------------------------------
void Function::eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                        values,
                    const Eigen::Ref<const EigenRowArrayXXd> x,
                    const mesh::Cell& cell) const
{
  assert(_function_space);
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();

  // FIXME: Should this throw an error instead?
  if (cell.mesh().id() != mesh.id())
  {
    eval(values, x);
    return;
  }

  assert(x.rows() == values.rows());
  assert(_function_space->element());
  const fem::FiniteElement& element = *_function_space->element();

  // Create work vector for expansion coefficients
  Eigen::Matrix<PetscScalar, 1, Eigen::Dynamic> coefficients(
      element.space_dimension());

  // Cell coordinates (re-allocated inside function for thread safety)
  EigenRowArrayXXd coordinate_dofs(cell.num_vertices(), mesh.geometry().dim());
  cell.get_coordinate_dofs(coordinate_dofs);

 // Restrict function to cell
  restrict(coefficients.data(), element, cell, coordinate_dofs);

  // Get coordinate mapping
  std::shared_ptr<const fem::CoordinateMapping> cmap
      = mesh.geometry().coord_mapping;
  if (!cmap)
  {
    throw std::runtime_error(
        "fem::CoordinateMapping has not been attached to mesh.");
  }

  std::size_t num_points = x.rows();
  std::size_t gdim = mesh.geometry().dim();
  std::size_t tdim = mesh.topology().dim();

  std::size_t reference_value_size = element.reference_value_size();
  std::size_t value_size = element.value_size();
  std::size_t space_dimension = element.space_dimension();

  Eigen::Tensor<double, 3, Eigen::RowMajor> J(num_points, gdim, tdim);
  EigenArrayXd detJ(num_points);
  Eigen::Tensor<double, 3, Eigen::RowMajor> K(num_points, tdim, gdim);

  EigenRowArrayXXd X(x.rows(), tdim);
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
      num_points, space_dimension, reference_value_size);

  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(
      num_points, space_dimension, value_size);

  // Compute reference coordinates X, and J, detJ and K
  cmap->compute_reference_geometry(X, J, detJ, K, x, coordinate_dofs);

  // Compute basis on reference element
  element.evaluate_reference_basis(basis_reference_values, X);

  // Push basis forward to physical element
  element.transform_reference_basis(basis_values, basis_reference_values, X, J,
                                    detJ, K);

  // Compute expansion
  values.setZero();
  for (std::size_t p = 0; p < num_points; ++p)
  {
    for (std::size_t i = 0; i < space_dimension; ++i)
    {
      for (std::size_t j = 0; j < value_size; ++j)
      {
        // TODO: Find an Eigen shortcut fot this operation
        values.row(p)[j] += coefficients[i] * basis_values(p, i, j);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(const GenericFunction& v)
{
  assert(_vector);
  assert(_function_space);

  // Interpolate
  _function_space->interpolate(*_vector, v);
}
//-----------------------------------------------------------------------------
std::size_t Function::value_rank() const
{
  assert(_function_space);
  assert(_function_space->element());
  return _function_space->element()->value_rank();
}
//-----------------------------------------------------------------------------
std::size_t Function::value_dimension(std::size_t i) const
{
  assert(_function_space);
  assert(_function_space->element());
  return _function_space->element()->value_dimension(i);
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> Function::value_shape() const
{
  assert(_function_space);
  assert(_function_space->element());
  std::vector<std::size_t> _shape(this->value_rank(), 1);
  for (std::size_t i = 0; i < _shape.size(); ++i)
    _shape[i] = this->value_dimension(i);
  return _shape;
}
//-----------------------------------------------------------------------------
void Function::restrict(
    PetscScalar* w, const fem::FiniteElement& element,
    const mesh::Cell& dolfin_cell,
    const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
{
  assert(w);
  assert(_function_space);
  assert(_function_space->dofmap());

  // Check if we are restricting to an element of this function space
  if (_function_space->has_element(element)
      && _function_space->has_cell(dolfin_cell))
  {
    // Get dofmap for cell
    const fem::GenericDofMap& dofmap = *_function_space->dofmap();
    auto dofs = dofmap.cell_dofs(dolfin_cell.index());

    // Note: We should have dofmap.max_element_dofs() == dofs.size() here.
    // Pick values from vector(s)
    _vector->get_local(w, dofs.size(), dofs.data());
  }
  else
    dolfin_not_implemented();

  //  {
  //    // Restrict as UFC function (by calling eval)
  //    element.evaluate_dofs(w, *this, coordinate_dofs, ufc_cell.orientation,
  //                          ufc_cell);
  //  }
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Function::compute_point_values(const mesh::Mesh& mesh) const
{
  assert(_function_space);
  assert(_function_space->mesh());

  // Check that the mesh matches. Notice that the hash is only
  // compared if the pointers are not matching.
  if (&mesh != _function_space->mesh().get()
      and mesh.hash() != _function_space->mesh()->hash())
  {
    throw std::runtime_error(
        "Cannot interpolate function values at points. Non-matching mesh");
  }

  // Local data for interpolation on each cell
  const std::size_t num_cell_vertices
      = mesh.type().num_vertices(mesh.topology().dim());

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  // Resize Array for holding point values
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      point_values(mesh.geometry().num_points(), value_size_loc);

  // Interpolate point values on each cell (using last computed value
  // if not continuous, e.g. discontinuous Galerkin methods)
  EigenRowArrayXXd x(num_cell_vertices, mesh.geometry().dim());
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values(num_cell_vertices, value_size_loc);

  const std::size_t tdim = mesh.topology().dim();
  const mesh::MeshConnectivity& cell_dofs
      = mesh.coordinate_dofs().entity_points(tdim);

  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Get coordinates for all points in cell
    cell.get_coordinate_dofs(x);
    values.resize(x.rows(), value_size_loc);

    // Call evaluate function
    eval(values, x, cell);

    // Copy values to array of point values
    const std::int32_t* dofs = cell_dofs(cell.index());
    for (unsigned int i = 0; i < x.rows(); ++i)
      point_values.row(dofs[i]) = values.row(i);
  }

  return point_values;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Function::compute_point_values() const
{
  assert(_function_space);
  assert(_function_space->mesh());
  return compute_point_values(*_function_space->mesh());
}
//-----------------------------------------------------------------------------
void Function::init_vector()
{
  common::Timer timer("Init dof vector");

  // Get dof map
  assert(_function_space);
  assert(_function_space->dofmap());
  const fem::GenericDofMap& dofmap = *(_function_space->dofmap());

  // Check that function space is not a subspace (view)
  if (dofmap.is_view())
  {
    std::runtime_error("Cannot initialize vector of degrees of freedom for "
                       "function. Cannot be created from subspace. Consider "
                       "collapsing the function space");
  }

  // Get index map
  std::shared_ptr<const common::IndexMap> index_map = dofmap.index_map();
  assert(index_map);

  _vector = std::make_shared<la::PETScVector>(*index_map);
  assert(_vector);
  _vector->set(0.0);
}
//-----------------------------------------------------------------------------
