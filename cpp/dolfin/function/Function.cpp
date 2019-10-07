// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Function.h"
#include <algorithm>
#include <cfloat>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/UniqueIdGenerator.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/utils.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/utils.h>
#include <dolfin/mesh/CoordinateDofs.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

using namespace dolfin;
using namespace dolfin::function;

namespace
{
//-----------------------------------------------------------------------------
// Create a vector with layout from dofmap, and zero.
la::PETScVector create_vector(const function::FunctionSpace& V)
{
  common::Timer timer("Init dof vector");

  // Get dof map
  assert(V.dofmap());
  const fem::DofMap& dofmap = *(V.dofmap());

  // Check that function space is not a subspace (view)
  assert(dofmap.element_dof_layout);
  if (dofmap.element_dof_layout->is_view())
  {
    std::runtime_error("Cannot initialize vector of degrees of freedom for "
                       "function. Cannot be created from subspace. Consider "
                       "collapsing the function space");
  }

  assert(dofmap.index_map);
  la::PETScVector v = la::PETScVector(*dofmap.index_map);
  la::VecWrapper _v(v.vec());
  _v.x.setZero();

  return v;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V)
    : _id(common::UniqueIdGenerator::id()), _function_space(V),
      _vector(create_vector(*V))
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
    : _id(common::UniqueIdGenerator::id()), _function_space(V), _vector(x)
{
  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  assert(V->dofmap());
  assert(V->dofmap()->index_map->size_global()
             * V->dofmap()->index_map->block_size
         <= _vector.size());
}
//-----------------------------------------------------------------------------
Function Function::sub(int i) const
{
  // Extract function subspace
  auto sub_space = _function_space->sub({i});

  // Return sub-function
  assert(sub_space);
  return Function(sub_space, _vector.vec());
}
//-----------------------------------------------------------------------------
Function Function::collapse() const
{
  // Create new collapsed FunctionSpace
  std::shared_ptr<const FunctionSpace> function_space_new;
  std::vector<PetscInt> collapsed_map;
  std::tie(function_space_new, collapsed_map) = _function_space->collapse();

  // Create new vector
  assert(function_space_new);
  la::PETScVector vector_new = create_vector(*function_space_new);

  // Wrap PETSc vectors using Eigen
  la::VecReadWrapper v_wrap(_vector.vec());
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x_old
      = v_wrap.x;
  la::VecWrapper v_new(vector_new.vec());
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x_new = v_new.x;

  // Copy values into new vector
  for (std::size_t i = 0; i < collapsed_map.size(); ++i)
  {
    assert((int)i < x_new.size());
    assert(collapsed_map[i] < x_old.size());
    x_new[i] = x_old[collapsed_map[i]];
  }

  return Function(function_space_new, vector_new.vec());
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace> Function::function_space() const
{
  return _function_space;
}
//-----------------------------------------------------------------------------
la::PETScVector& Function::vector()
{
  // Check that this is not a sub function.
  assert(_function_space->dofmap());
  assert(_function_space->dofmap()->index_map);
  if (_vector.size()
      != _function_space->dofmap()->index_map->size_global()
             * _function_space->dofmap()->index_map->block_size)
  {
    throw std::runtime_error(
        "Cannot access a non-const vector from a subfunction");
  }

  return _vector;
}
//-----------------------------------------------------------------------------
const la::PETScVector& Function::vector() const { return _vector; }
//-----------------------------------------------------------------------------
void Function::eval(
    const Eigen::Ref<
        const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& x,
    const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& cells,
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        u) const
{
  // TODO: This could be easily made more efficient by exploiting points
  // being ordered by the cell to which they belong.

  if (x.rows() != cells.rows())
  {
    throw std::runtime_error(
        "Number of points and number of cells must be equal.");
  }
  if (x.rows() != u.rows())
  {
    throw std::runtime_error("Length of array for Function values must be the "
                             "same as the number of points.");
  }

  // Get mesh
  assert(_function_space);
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Get geometry data
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.entity_positions();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.connections();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Get coordinate mapping
  std::shared_ptr<const fem::CoordinateMapping> cmap
      = mesh.geometry().coord_mapping;
  if (!cmap)
  {
    throw std::runtime_error(
        "fem::CoordinateMapping has not been attached to mesh.");
  }

  // Get element
  assert(_function_space->element());
  const fem::FiniteElement& element = *_function_space->element();
  const int reference_value_size = element.reference_value_size();
  const int value_size = element.value_size();
  const int space_dimension = element.space_dimension();

  // Prepare geometry data structures
  Eigen::Tensor<double, 3, Eigen::RowMajor> J(1, gdim, tdim);
  Eigen::Array<double, Eigen::Dynamic, 1> detJ(1);
  Eigen::Tensor<double, 3, Eigen::RowMajor> K(1, tdim, gdim);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(1,
                                                                          tdim);

  // Prepare basis function data structures
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
      1, space_dimension, reference_value_size);
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(1, space_dimension,
                                                         value_size);

  // Create work vector for expansion coefficients
  Eigen::Matrix<PetscScalar, 1, Eigen::Dynamic> coefficients(
      element.space_dimension());

  // Get dofmap
  assert(_function_space->dofmap());
  const fem::DofMap& dofmap = *_function_space->dofmap();

  // Loop over points
  u.setZero();
  la::VecReadWrapper v(_vector.vec());
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _v = v.x;
  for (Eigen::Index p = 0; p < cells.rows(); ++p)
  {
    const int cell_index = cells(p);

    // Skip negative cell indices
    if (cell_index < 0)
      break;

    // Get cell geometry (coordinate dofs)
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Compute reference coordinates X, and J, detJ and K
    cmap->compute_reference_geometry(X, J, detJ, K, x.row(p).head(gdim),
                                     coordinate_dofs);

    // Compute basis on reference element
    element.evaluate_reference_basis(basis_reference_values, X);

    // Push basis forward to physical element
    element.transform_reference_basis(basis_values, basis_reference_values, X,
                                      J, detJ, K);

    // Get degrees of freedom for current cell
    auto dofs = dofmap.cell_dofs(cell_index);
    for (Eigen::Index i = 0; i < dofs.size(); ++i)
      coefficients[i] = _v[dofs[i]];

    // Compute expansion
    for (int i = 0; i < space_dimension; ++i)
    {
      for (int j = 0; j < value_size; ++j)
      {
        // TODO: Find an Eigen shortcut for this operation
        u.row(p)[j] += coefficients[i] * basis_values(0, i, j);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void Function::eval_reference(
    const Eigen::Ref<
        const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& X,
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        u) const
{
  // Get mesh
  assert(_function_space);
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  if (tdim != gdim)
  {
    throw std::runtime_error("Function evaluation at reference cell "
                             "coordinates is not yet supported for manifolds.");
  }

  // Get element
  assert(_function_space->element());
  const fem::FiniteElement& element = *_function_space->element();
  const int reference_value_size = element.reference_value_size();
  const int value_size = element.value_size();
  const int space_dimension = element.space_dimension();

  // Compute basis on reference element

  // NOTE: This step needs to be generalised for elements other than
  //       non-manifold Lagrange elements
  Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
      X.rows(), space_dimension, reference_value_size);
  element.evaluate_reference_basis(basis_reference_values, X.leftCols(gdim));

  // Get dofmap
  assert(_function_space->dofmap());
  const fem::DofMap& dofmap = *_function_space->dofmap();

  // Unwrap data vector
  la::VecReadWrapper v(_vector.vec());
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _v = v.x;

  // Loop over cells
  u.setZero();
  for (int c = 0; c < mesh.num_entities(tdim); ++c)
  {
    auto dofs = dofmap.cell_dofs(c);

    // Loop over points
    for (int p = 0; p < X.rows(); ++p)
    {
      for (int i = 0; i < space_dimension; ++i)
      {
        for (int j = 0; j < value_size; ++j)
        {
          // TODO: Find an Eigen shortcut for this operation
          u.row(c * X.rows() + p)[j]
              += _v[dofs[i]] * basis_reference_values(p, i, j);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(const Function& v)
{
  assert(_function_space);
  la::VecWrapper x(_vector.vec());
  _function_space->interpolate(x.x, v);
}
//-----------------------------------------------------------------------------
void Function::interpolate(const FunctionSpace::interpolation_function& f)
{
  la::VecWrapper x(_vector.vec());
  _function_space->interpolate(x.x, f);
}
//-----------------------------------------------------------------------------
int Function::value_rank() const
{
  assert(_function_space);
  assert(_function_space->element());
  return _function_space->element()->value_rank();
}
//-----------------------------------------------------------------------------
int Function::value_size() const
{
  int size = 1;
  for (int i = 0; i < value_rank(); ++i)
    size *= value_dimension(i);
  return size;
}
//-----------------------------------------------------------------------------
int Function::value_dimension(int i) const
{
  assert(_function_space);
  assert(_function_space->element());
  return _function_space->element()->value_dimension(i);
}
//-----------------------------------------------------------------------------
std::vector<int> Function::value_shape() const
{
  std::vector<int> _shape(this->value_rank(), 1);
  for (std::size_t i = 0; i < _shape.size(); ++i)
    _shape[i] = this->value_dimension(i);
  return _shape;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Function::compute_point_values() const
{
  assert(_function_space);
  assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();

  const int tdim = mesh.topology().dim();

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  // Resize Array for holding point values
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      point_values(mesh.geometry().num_points(), value_size_loc);

  const mesh::Connectivity& cell_dofs = mesh.coordinate_dofs().entity_points();

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.entity_positions();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Interpolate point values on each cell (using last computed value if
  // not continuous, e.g. discontinuous Galerkin methods)
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x(num_dofs_g, 3);
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values(num_dofs_g, value_size_loc);
  for (auto& cell : mesh::MeshRange(mesh, tdim, mesh::MeshRangeType::ALL))
  {
    // Get coordinates for all points in cell
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      x.row(i) = x_g.row(cell_g[pos_g[cell_index] + i]);

    values.resize(x.rows(), value_size_loc);

    // Call evaluate function
    Eigen::Array<int, Eigen::Dynamic, 1> cells(x.rows());
    cells = cell.index();
    eval(x, cells, values);

    // Copy values to array of point values
    const std::int32_t* dofs = cell_dofs.connections(cell.index());
    for (int i = 0; i < x.rows(); ++i)
      point_values.row(dofs[i]) = values.row(i);
  }

  return point_values;
}
//-----------------------------------------------------------------------------
std::size_t Function::id() const { return _id; }
//-----------------------------------------------------------------------------
