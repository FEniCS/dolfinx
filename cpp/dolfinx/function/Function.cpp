// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Function.h"
#include "FunctionSpace.h"
#include <cfloat>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::function;

namespace
{
//-----------------------------------------------------------------------------
// Create a vector with layout from dofmap, and zero.
// la::PETScVector create_vector(const function::FunctionSpace& V)
// {
//   common::Timer timer("Init dof vector");

//   // Get dof map
//   std::shared_ptr<const fem::DofMap> dofmap = V.dofmap();
//   assert(dofmap);

//   // Check that function space is not a subspace (view)
//   assert(dofmap->element_dof_layout);
//   if (dofmap->element_dof_layout->is_view())
//   {
//     throw std::runtime_error(
//         "Cannot initialize vector of degrees of freedom for "
//         "function. Cannot be created from subspace. Consider "
//         "collapsing the function space");
//   }

//   assert(dofmap->index_map);
//   la::PETScVector v = la::PETScVector(*(dofmap->index_map));
//   la::VecWrapper _v(v.vec());
//   _v.x.setZero();

//   return v;
// }
//-----------------------------------------------------------------------------
Vec create_ghosted_vector(const common::IndexMap& map,
                          Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> x)
{
  const int bs = map.block_size();
  std::int32_t size_local = bs * map.size_local();
  std::int32_t num_ghosts = bs * map.num_ghosts();
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts = map.ghosts();
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _ghosts(bs * ghosts.rows());
  for (int i = 0; i < ghosts.rows(); ++i)
  {
    for (int j = 0; j < bs; ++j)
      _ghosts[i * bs + j] = bs * ghosts[i] + j;
  }

  Vec vec;
  VecCreateGhostWithArray(map.comm(), size_local, PETSC_DECIDE, num_ghosts,
                          _ghosts.data(), x.array().data(), &vec);
  return vec;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V)
    : _id(common::UniqueIdGenerator::id()), _function_space(V),
      _x(std::make_shared<la::Vector<PetscScalar>>(V->dofmap()->index_map)),
      _vector(create_ghosted_vector(*V->dofmap()->index_map, _x->array()),
              false)
// _vector(create_vector(*V))
{
  if (!V->component().empty())
  {
    throw std::runtime_error("Cannot create Function from subspace. Consider "
                             "collapsing the function space");
  }

  _x->array().setZero();

  // auto map = V->dofmap()->index_map;
  // const int bs = map->block_size();
  // std::int32_t size_local = bs * map->size_local();
  // std::int32_t num_ghosts = bs * map->num_ghosts();
  // const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts =
  // map->ghosts(); Eigen::Array<PetscInt, Eigen::Dynamic, 1> _ghosts(bs *
  // ghosts.rows()); for (int i = 0; i < ghosts.rows(); ++i)
  // {
  //   for (int j = 0; j < bs; ++j)
  //     _ghosts[i * bs + j] = bs * ghosts[i] + j;
  // }

  // Vec test;
  // VecCreateGhostWithArray(map->comm(), size_local, PETSC_DECIDE, num_ghosts,
  //                         _ghosts.data(), _x.array().data(), &test);
  // la::PETScVector x(test, false);
  // _vector = std::move(x);
}
//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V,
                   std::shared_ptr<la::Vector<PetscScalar>> x)
    : _id(common::UniqueIdGenerator::id()), _function_space(V), _x(x),
      _vector(create_ghosted_vector(*V->dofmap()->index_map, _x->array()),
              false)
{
  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  assert(V->dofmap());
  assert(V->dofmap()->index_map->size_global()
             * V->dofmap()->index_map->block_size()
         <= _vector.size());
}
//-----------------------------------------------------------------------------
Function Function::sub(int i) const
{
  // Extract function subspace
  auto sub_space = _function_space->sub({i});

  // Return sub-function
  assert(sub_space);
  return Function(sub_space, _x);
}
//-----------------------------------------------------------------------------
Function Function::collapse() const
{
  // Create new collapsed FunctionSpace
  const auto [function_space_new, collapsed_map] = _function_space->collapse();

  // Create new vector
  assert(function_space_new);
  auto vector_new = std::make_shared<la::Vector<PetscScalar>>(
      function_space_new->dofmap()->index_map);

  // la::PETScVector vector_new = create_vector(*function_space_new);

  // // Wrap PETSc vectors using Eigen
  // la::VecReadWrapper v_wrap(_vector.vec());
  // Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x_old
  //     = v_wrap.x;
  // la::VecWrapper v_new(vector_new.vec());
  // Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x_new = v_new.x;

  const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>& x_old = _x->array();
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>& x_new = vector_new->array();

  // Copy values into new vector
  for (std::size_t i = 0; i < collapsed_map.size(); ++i)
  {
    assert((int)i < x_new.size());
    assert(collapsed_map[i] < x_old.size());
    x_new[i] = x_old[collapsed_map[i]];
  }

  return Function(function_space_new, vector_new);
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
             * _function_space->dofmap()->index_map->block_size())
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
  std::shared_ptr<const mesh::Mesh> mesh = _function_space->mesh();
  assert(mesh);
  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Get geometry data
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Get coordinate map
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Get element
  assert(_function_space->element());
  std::shared_ptr<const fem::FiniteElement> element
      = _function_space->element();
  assert(element);
  const int reference_value_size = element->reference_value_size();
  const int value_size = element->value_size();
  const int space_dimension = element->space_dimension();

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
  Eigen::Matrix<PetscScalar, 1, Eigen::Dynamic> coefficients(space_dimension);

  // Get dofmap
  std::shared_ptr<const fem::DofMap> dofmap = _function_space->dofmap();
  assert(dofmap);

  mesh->topology_mutable().create_entity_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh->topology().get_cell_permutation_info();

  // Loop over points
  u.setZero();
  la::VecReadWrapper v(_vector.vec());
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _v = v.x;
  for (Eigen::Index p = 0; p < cells.rows(); ++p)
  {
    const int cell_index = cells(p);

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(cell_index);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Compute reference coordinates X, and J, detJ and K
    cmap.compute_reference_geometry(X, J, detJ, K, x.row(p).head(gdim),
                                    coordinate_dofs);

    // Compute basis on reference element
    element->evaluate_reference_basis(basis_reference_values, X);

    // Push basis forward to physical element
    element->transform_reference_basis(basis_values, basis_reference_values, X,
                                       J, detJ, K, cell_info[cell_index]);

    // Get degrees of freedom for current cell
    auto dofs = dofmap->cell_dofs(cell_index);
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
void Function::interpolate(const Function& v)
{
  assert(_function_space);
  la::VecWrapper x(_vector.vec());
  _function_space->interpolate(x.x, v);
}
//-----------------------------------------------------------------------------
void Function::interpolate(
    const std::function<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& f)
{
  la::VecWrapper x(_vector.vec());
  _function_space->interpolate(x.x, f);
}
//-----------------------------------------------------------------------------
void Function::interpolate_c(const FunctionSpace::interpolation_function& f)
{
  la::VecWrapper x(_vector.vec());
  _function_space->interpolate_c(x.x, f);
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Function::compute_point_values() const
{
  assert(_function_space);
  std::shared_ptr<const mesh::Mesh> mesh = _function_space->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  // Compute in tensor (one for scalar function, . . .)
  const int value_size_loc = _function_space->element()->value_size();

  // Resize Array for holding point values
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      point_values(mesh->geometry().x().rows(), value_size_loc);

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh->geometry().x();

  // Interpolate point values on each cell (using last computed value if
  // not continuous, e.g. discontinuous Galerkin methods)
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x(num_dofs_g, 3);
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values(num_dofs_g, value_size_loc);
  auto map = mesh->topology().index_map(tdim);
  assert(map);
  const std::int32_t num_cells = map->size_local() + map->num_ghosts();
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Get coordinates for all points in cell
    auto dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      x.row(i) = x_g.row(dofs[i]);

    values.resize(x.rows(), value_size_loc);

    // Call evaluate function
    Eigen::Array<int, Eigen::Dynamic, 1> cells(x.rows());
    cells = c;
    eval(x, cells, values);

    // Copy values to array of point values
    for (int i = 0; i < x.rows(); ++i)
      point_values.row(dofs[i]) = values.row(i);
  }

  return point_values;
}
//-----------------------------------------------------------------------------
std::size_t Function::id() const { return _id; }
//-----------------------------------------------------------------------------
