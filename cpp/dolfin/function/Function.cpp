// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Function.h"
#include "FunctionSpace.h"
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
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/utils.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
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
    : id(common::UniqueIdGenerator::id()), _function_space(V),
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
    : id(common::UniqueIdGenerator::id()), _function_space(V), _vector(x)
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
  assert(_function_space);
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
  const int gdim = x.cols();
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  for (unsigned int i = 0; i < x.rows(); ++i)
  {
    // Pad the input point to size 3 (bounding box requires 3d point)
    point.head(gdim) = x.row(i);

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
  // FIXME: This function needs to be changed to handle an arbitrary
  // number of points for efficiency

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

  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Cell coordinates (re-allocated inside function for thread safety)
  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();
  EigenRowArrayXXd coordinate_dofs(num_dofs_g, gdim);

  const int cell_index = cell.index();
  for (int i = 0; i < num_dofs_g; ++i)
    for (int j = 0; j < gdim; ++j)
      coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

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
  la::VecWrapper x(_vector.vec());
  _function_space->interpolate(x.x, v);
}
//-----------------------------------------------------------------------------
void Function::interpolate(
    const std::function<
        void(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>>,
             const Eigen::Ref<const Eigen::Array<
                 double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>)>& f)

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
int Function::value_dimension(int i) const
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
  const fem::DofMap& dofmap = *_function_space->dofmap();
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

  // Check that the mesh matches. Notice that the hash is only compared
  // if the pointers are not matching.
  if (&mesh != _function_space->mesh().get()
      and mesh.hash() != _function_space->mesh()->hash())
  {
    throw std::runtime_error(
        "Cannot interpolate function values at points. Non-matching mesh");
  }

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  // Resize Array for holding point values
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      point_values(mesh.geometry().num_points(), value_size_loc);

  const int gdim = mesh.topology().dim();
  const mesh::Connectivity& cell_dofs = mesh.coordinate_dofs().entity_points();

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Interpolate point values on each cell (using last computed value
  // if not continuous, e.g. discontinuous Galerkin methods)
  EigenRowArrayXXd x(num_dofs_g, mesh.geometry().dim());
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values(num_dofs_g, value_size_loc);
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Get coordinates for all points in cell
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        x(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    values.resize(x.rows(), value_size_loc);

    // Call evaluate function
    eval(values, x, cell);

    // Copy values to array of point values
    const std::int32_t* dofs = cell_dofs.connections(cell.index());
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
std::size_t Function::value_size() const
{
  std::size_t size = 1;
  for (int i = 0; i < value_rank(); ++i)
    size *= value_dimension(i);
  return size;
}
//-----------------------------------------------------------------------------
