// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Function.h"
#include "Expression.h"
#include "FunctionSpace.h"
#include <algorithm>
#include <cfloat>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <unordered_map>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V)
    : common::Variable("u"), _function_space(V), _vector(_create_vector(*V))
{
  // Check that we don't have a subspace
  if (!V->component().empty())
  {
    throw std::runtime_error("Cannot create Function from subspace. Consider "
                             "collapsing the function space");
  }
}
//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V, Vec x)
    : _function_space(V), _vector(x)
{
  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  assert(V->dofmap());
  assert(V->dofmap()->global_dimension() <= _vector.size());
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v) : _vector(v.vector().vec())
{
  // Make a copy of all the data, or if v is a sub-function, then we
  // collapse the dof map and copy only the relevant entries from the
  // vector of v.
  if (v._vector.size() == v._function_space->dim())
  {
    // Copy function space pointer
    this->_function_space = v._function_space;

    // Copy vector
    this->_vector = v._vector.copy();
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
    la::VecReadWrapper v_wrap(v.vector().vec());
    Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x
        = v_wrap.x;
    std::vector<PetscScalar> gathered_values(collapsed_map.size());
    for (std::size_t j = 0; j < gathered_values.size(); ++j)
      gathered_values[j] = x[old_rows[j]];

    // Initial new vector (global)
    _vector = _create_vector(*_function_space);
    assert(_function_space->dofmap());
    assert(_vector.size() == _function_space->dofmap()->global_dimension());

    // FIXME (local): Check this for local or global
    // Set values in vector
    la::VecWrapper v(this->_vector.vec());
    for (std::size_t i = 0; i < collapsed_map.size(); ++i)
      v.x[new_rows[i]] = gathered_values[i];
    v.restore();
  }
}
//-----------------------------------------------------------------------------
Function Function::sub(std::size_t i) const
{
  // Extract function subspace
  auto sub_space = _function_space->sub({i});

  // Return sub-function
  assert(sub_space);
  return Function(sub_space, _vector.vec());
}
//-----------------------------------------------------------------------------
la::PETScVector& Function::vector()
{
  assert(_function_space->dofmap());

  // Check that this is not a sub function.
  if (_vector.size() != _function_space->dofmap()->global_dimension())
  {
    throw std::runtime_error(
        "Cannot access a non-const vector from a subfunction");
  }

  return _vector;
}
//-----------------------------------------------------------------------------
const la::PETScVector& Function::vector() const { return _vector; }
//-----------------------------------------------------------------------------
void Function::eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>
                        values,
                    const Eigen::Ref<const EigenRowArrayXXd> x,
                    const geometry::BoundingBoxTree& bb_tree) const
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
    unsigned int id = bb_tree.compute_first_entity_collision(point, mesh);

    // If not found, use the closest cell
    if (id == std::numeric_limits<unsigned int>::max())
    {
      // Check if the closest cell is within 2*DBL_EPSILON. This we can
      // allow without _allow_extrapolation
      std::pair<unsigned int, double> close
          = bb_tree.compute_closest_entity(point, mesh);

      if (close.second < 2.0 * DBL_EPSILON)
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
void Function::eval(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>
        x,
    const mesh::Cell& cell) const
{
  assert(_function_space);
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();
  if (cell.mesh().id() != mesh.id())
  {
    throw std::runtime_error(
        "Cell passed to Function::eval is from a different mesh.");
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

  restrict(coefficients.data(), cell, coordinate_dofs);

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
void Function::interpolate(const Function& v)
{
  assert(_function_space);
  _function_space->interpolate(_vector, v);
}
//-----------------------------------------------------------------------------
void Function::interpolate(const Expression& expr)
{
  assert(_function_space);
  _function_space->interpolate(_vector, expr);
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
    PetscScalar* w, const mesh::Cell& dolfin_cell,
    const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
{
  assert(w);
  assert(_function_space);
  assert(_function_space->dofmap());

  // Get dofmap for cell
  const fem::GenericDofMap& dofmap = *_function_space->dofmap();
  auto dofs = dofmap.cell_dofs(dolfin_cell.index());

  // Pick values from vector(s)
  la::VecReadWrapper v(_vector.vec());
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _v = v.x;
  for (Eigen::Index i = 0; i < dofs.size(); ++i)
    w[i] = _v[dofs[i]];
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
la::PETScVector Function::_create_vector(const function::FunctionSpace& V)
{
  common::Timer timer("Init dof vector");

  // Get dof map
  assert(V.dofmap());
  const fem::GenericDofMap& dofmap = *(V.dofmap());

  // Check that function space is not a subspace (view)
  if (dofmap.is_view())
  {
    std::runtime_error("Cannot initialize vector of degrees of freedom for "
                       "function. Cannot be created from subspace. Consider "
                       "collapsing the function space");
  }

  assert(dofmap.index_map());
  la::PETScVector v = la::PETScVector(*dofmap.index_map());
  VecSet(v.vec(), 0.0);

  return v;
}
//-----------------------------------------------------------------------------
std::size_t Function::value_size() const
{
  std::size_t size = 1;
  for (std::size_t i = 0; i < value_rank(); ++i)
    size *= value_dimension(i);
  return size;
}
//-----------------------------------------------------------------------------
