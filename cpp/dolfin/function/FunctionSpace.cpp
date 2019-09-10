// Copyright (C) 2008-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FunctionSpace.h"
#include "Function.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/UniqueIdGenerator.h>
#include <dolfin/common/types.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/CoordinateDofs.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>
#include <vector>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::shared_ptr<const mesh::Mesh> mesh,
                             std::shared_ptr<const fem::FiniteElement> element,
                             std::shared_ptr<const fem::DofMap> dofmap)
    : mesh(mesh), element(element), dofmap(dofmap),
      id(common::UniqueIdGenerator::id()), _root_space_id(id)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator==(const FunctionSpace& V) const
{
  // Compare pointers to shared objects
  return element.get() == V.element.get() and mesh.get() == V.mesh.get()
         and dofmap.get() == V.dofmap.get();
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator!=(const FunctionSpace& V) const
{
  // Compare pointers to shared objects
  return !(*this == V);
}
//-----------------------------------------------------------------------------
std::int64_t FunctionSpace::dim() const
{
  assert(dofmap);
  assert(dofmap->element_dof_layout);
  if (dofmap->element_dof_layout->is_view())
  {
    throw std::runtime_error("FunctionSpace dimension not supported for "
                             "sub-functions");
  }

  assert(dofmap->index_map);
  return dofmap->index_map->size_global() * dofmap->index_map->block_size;
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate_from_any(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        expansion_coefficients,
    const Function& v) const
{
  if (!v.function_space()->has_element(*element))
  {
    throw std::runtime_error("Restricting finite elements function in "
                             "different elements not suppoted.");
  }

  assert(mesh);
  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Initialize local arrays
  assert(dofmap->element_dof_layout);
  std::vector<PetscScalar> cell_coefficients(
      dofmap->element_dof_layout->num_dofs());

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh->coordinate_dofs().entity_points();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = mesh->geometry().points();

  // Iterate over mesh and interpolate on each cell
  EigenRowArrayXXd coordinate_dofs(num_dofs_g, gdim);
  for (auto& cell : mesh::MeshRange(*mesh, tdim))
  {
    // FIXME: Move this out
    if (!v.function_space()->has_cell(cell))
    {
      throw std::runtime_error("Restricting finite elements function in "
                               "different elements not suppoted.");
    }

    // Get cell coordinate dofs
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Restrict function to cell
    v.restrict(cell, coordinate_dofs, cell_coefficients.data());

    // Tabulate dofs
    auto cell_dofs = dofmap->cell_dofs(cell.index());

    for (Eigen::Index i = 0; i < cell_dofs.size(); ++i)
      expansion_coefficients[cell_dofs[i]] = cell_coefficients[i];
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        expansion_coefficients,
    const Function& v) const
{
  assert(mesh);
  assert(element);
  assert(dofmap);

  // Check that function ranks match
  if (element->value_rank() != v.value_rank())
  {
    throw std::runtime_error("Cannot interpolate function into function space. "
                             "Rank of function ("
                             + std::to_string(v.value_rank())
                             + ") does not match rank of function space ("
                             + std::to_string(element->value_rank()) + ")");
  }

  // Check that function dimension match
  for (int i = 0; i < element->value_rank(); ++i)
  {
    if (element->value_dimension(i) != v.value_dimension(i))
    {
      throw std::runtime_error(
          "Cannot interpolate function into function space. "
          "Dimension "
          + std::to_string(i) + " of function ("
          + std::to_string(v.value_dimension(i)) + ") does not match dimension "
          + std::to_string(i) + " of function space("
          + std::to_string(element->value_dimension(i)) + ")");
    }
  }

  std::shared_ptr<const FunctionSpace> v_fs = v.function_space();
  interpolate_from_any(expansion_coefficients, v);
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        expansion_coefficients,
    const std::function<void(
        Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>,
        const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>)>&
        f) const
{
  assert(mesh);
  assert(element);
  assert(dofmap);
  const int tdim = mesh->topology().dim();

  // Note: the following does not exploit any block structure, e.g. for
  // vector Lagrange, which leads to a lot of redundant evaluations.
  // E.g., for a vector Lagrange element the vector-valued expression is
  // evaluted three times at the some point.

  // Build list of points at which to evaluate the Expression
  EigenRowArrayXXd x = tabulate_dof_coordinates();

  // Evaluate Expression at points
  // std::vector<int> vshape = e.value_shape();
  std::vector<int> vshape(element->value_rank(), 1);
  for (std::size_t i = 0; i < vshape.size(); ++i)
    vshape[i] = element->value_dimension(i);
  const int value_size = std::accumulate(std::begin(vshape), std::end(vshape),
                                         1, std::multiplies<>());
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values(x.rows(), value_size);
  assert(values.rows() == x.rows());
  f(values, x);

  // FIXME: Dummy coordinate dofs - should limit the interpolation to
  // Lagrange, in which case we don't need coordinate dofs in
  // FiniteElement::transform_values.
  EigenRowArrayXXd coordinate_dofs;

  // FIXME: It would be far more elegant and efficient to avoid the need
  // to loop over cells to set the expansion corfficients. Would be much
  // better if the expansion coefficients could be passed straight into
  // Expresion::eval.

  // Loop over cells
  const int ndofs = element->space_dimension();
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      values_cell(ndofs, value_size);
  assert(dofmap->element_dof_layout);
  std::vector<PetscScalar> cell_coefficients(
      dofmap->element_dof_layout->num_dofs());
  for (auto& cell : mesh::MeshRange(*mesh, tdim))
  {
    // Get dofmap for cell
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> cell_dofs
        = dofmap->cell_dofs(cell.index());
    for (Eigen::Index i = 0; i < cell_dofs.rows(); ++i)
    {
      for (Eigen::Index j = 0; j < value_size; ++j)
        values_cell(i, j) = values(cell_dofs[i], j);

      // FIXME: For vector-valued Lagrange, this function 'throws away'
      // the redundant expression evaluations. It should really be made
      // not necessary.
      element->transform_values(cell_coefficients.data(), values_cell,
                                coordinate_dofs);

      // Copy into expansion coefficient array
      for (Eigen::Index i = 0; i < cell_dofs.rows(); ++i)
        expansion_coefficients[cell_dofs[i]] = cell_coefficients[i];
    }
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace>
FunctionSpace::sub(const std::vector<int>& component) const
{
  assert(mesh);
  assert(element);
  assert(dofmap);

  // Check if sub space is already in the cache and not expired
  auto subspace_it = _subspaces.find(component);
  if (subspace_it != _subspaces.end())
  {
    if (auto s = subspace_it->second.lock())
      return s;
  }

  // Extract sub-element
  std::shared_ptr<const fem::FiniteElement> element
      = this->element->extract_sub_element(component);

  // Extract sub dofmap
  auto dofmap = std::make_shared<fem::DofMap>(
      this->dofmap->extract_sub_dofmap(component, *mesh));

  // Create new sub space
  auto sub_space = std::make_shared<FunctionSpace>(mesh, element, dofmap);

  // Set root space id and component w.r.t. root
  sub_space->_root_space_id = _root_space_id;
  sub_space->_component = _component;
  sub_space->_component.insert(sub_space->_component.end(), component.begin(),
                               component.end());

  // Insert new subspace into cache
  _subspaces.insert(std::pair<std::vector<int>, std::shared_ptr<FunctionSpace>>(
      sub_space->_component, sub_space));

  return sub_space;
}
//-----------------------------------------------------------------------------
std::pair<std::shared_ptr<FunctionSpace>, std::vector<PetscInt>>
FunctionSpace::collapse() const
{
  assert(mesh);
  if (_component.empty())
    throw std::runtime_error("Function space is not a subspace");

  // Create collapsed DofMap
  std::shared_ptr<fem::DofMap> collapsed_dofmap;
  std::vector<PetscInt> collapsed_dofs;
  std::tie(collapsed_dofmap, collapsed_dofs) = dofmap->collapse(*mesh);

  // Create new FunctionSpace and return
  auto collapsed_sub_space
      = std::make_shared<FunctionSpace>(mesh, element, collapsed_dofmap);

  return std::make_pair(std::move(collapsed_sub_space),
                        std::move(collapsed_dofs));
}
//-----------------------------------------------------------------------------
std::vector<int> FunctionSpace::component() const { return _component; }
//-----------------------------------------------------------------------------
EigenRowArrayXXd FunctionSpace::tabulate_dof_coordinates() const
{
  // Geometric dimension
  assert(mesh);
  assert(element);
  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  if (!_component.empty())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  // Get local size
  // Get local size
  assert(dofmap);
  std::shared_ptr<const common::IndexMap> index_map = dofmap->index_map;
  assert(index_map);
  std::size_t bs = index_map->block_size;
  std::size_t local_size
      = bs * (index_map->size_local() + index_map->num_ghosts());

  // Dof coordinate on reference element
  const EigenRowArrayXXd& X = element->dof_reference_coordinates();

  // Arrray to hold coordinates and return
  EigenRowArrayXXd x(local_size, gdim);

  // Get coordinate mapping
  if (!mesh->geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateMapping has not been attached to mesh.");
  }
  const fem::CoordinateMapping& cmap = *mesh->geometry().coord_mapping;

  // Cell coordinates (re-allocated inside function for thread safety)
  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh->coordinate_dofs().entity_points();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = mesh->geometry().points();

  // Loop over cells and tabulate dofs
  EigenRowArrayXXd coordinates(element->space_dimension(), gdim);
  EigenRowArrayXXd coordinate_dofs(num_dofs_g, gdim);
  for (auto& cell : mesh::MeshRange(*mesh, tdim))
  {
    // Update cell
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Get local-to-global map
    auto dofs = dofmap->cell_dofs(cell.index());

    // Tabulate dof coordinates on cell
    cmap.compute_physical_coordinates(coordinates, X, coordinate_dofs);

    // Copy dof coordinates into vector
    for (Eigen::Index i = 0; i < dofs.size(); ++i)
    {
      const PetscInt dof = dofs[i];
      if (dof < (PetscInt)local_size)
        x.row(dof) = coordinates.row(i);
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
void FunctionSpace::set_x(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x,
    PetscScalar value, int component) const
{
  assert(mesh);
  assert(dofmap);
  assert(element);

  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();
  std::vector<PetscScalar> x_values;

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh->coordinate_dofs().entity_points();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = mesh->geometry().points();

  // Dof coordinate on reference element
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X
      = element->dof_reference_coordinates();

  // Get coordinate mapping
  if (!mesh->geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateMapping has not been attached to mesh.");
  }
  const fem::CoordinateMapping& cmap = *mesh->geometry().coord_mapping;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinates(element->space_dimension(), mesh->geometry().dim());
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  for (auto& cell : mesh::MeshRange(*mesh, tdim))
  {
    // Update UFC cell
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Get cell local-to-global map
    auto dofs = dofmap->cell_dofs(cell.index());

    // Tabulate dof coordinates
    cmap.compute_physical_coordinates(coordinates, X, coordinate_dofs);

    assert(coordinates.rows() == dofs.size());
    assert(component < (int)coordinates.cols());

    // Copy coordinate (it may be possible to avoid this)
    for (Eigen::Index i = 0; i < coordinates.rows(); ++i)
      x[dofs[i]] = value * coordinates(i, component);
  }
}
//-----------------------------------------------------------------------------
bool FunctionSpace::contains(const FunctionSpace& V) const
{
  // Is the root space same?
  if (_root_space_id != V._root_space_id)
    return false;

  // Is V possibly our superspace?
  if (_component.size() > V._component.size())
    return false;

  // Are our components same as leading components of V?
  for (std::size_t i = 0; i < _component.size(); ++i)
  {
    if (_component[i] != V._component[i])
      return false;
  }

  // Ok, V is really our subspace
  return true;
}
//-----------------------------------------------------------------------------
