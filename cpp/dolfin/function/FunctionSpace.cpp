// Copyright (C) 2008-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FunctionSpace.h"
#include "Expression.h"
#include "Function.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <vector>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::shared_ptr<const mesh::Mesh> mesh,
                             std::shared_ptr<const fem::FiniteElement> element,
                             std::shared_ptr<const fem::GenericDofMap> dofmap)
    : _mesh(mesh), _element(element), _dofmap(dofmap), _root_space_id(id())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::shared_ptr<const mesh::Mesh> mesh)
    : _mesh(mesh), _root_space_id(id())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(const FunctionSpace& V)
{
  // Assign data (will be shared)
  *this = V;
}
//-----------------------------------------------------------------------------
FunctionSpace::~FunctionSpace()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FunctionSpace::attach(std::shared_ptr<const fem::FiniteElement> element,
                           std::shared_ptr<const fem::GenericDofMap> dofmap)
{
  _element = element;
  _dofmap = dofmap;
}
//-----------------------------------------------------------------------------
const FunctionSpace& FunctionSpace::operator=(const FunctionSpace& V)
{
  // Assign data (will be shared)
  _mesh = V._mesh;
  _element = V._element;
  _dofmap = V._dofmap;
  _component = V._component;

  // Call assignment operator for base class
  common::Variable::operator=(V);

  return *this;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator==(const FunctionSpace& V) const
{
  // Compare pointers to shared objects
  return _element.get() == V._element.get() && _mesh.get() == V._mesh.get()
         && _dofmap.get() == V._dofmap.get();
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator!=(const FunctionSpace& V) const
{
  // Compare pointers to shared objects
  return !(*this == V);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::Mesh> FunctionSpace::mesh() const { return _mesh; }
//-----------------------------------------------------------------------------
std::shared_ptr<const fem::FiniteElement> FunctionSpace::element() const
{
  return _element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const fem::GenericDofMap> FunctionSpace::dofmap() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
std::int64_t FunctionSpace::dim() const
{
  assert(_dofmap);
  return _dofmap->global_dimension();
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate_from_any(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        expansion_coefficients,
    const Function& v) const
{
  assert(_mesh);
  const int gdim = _mesh->geometry().dim();
  const int tdim = _mesh->topology().dim();

  // Initialize local arrays
  std::vector<PetscScalar> cell_coefficients(_dofmap->max_element_dofs());

  if (!v.function_space()->has_element(*_element))
  {
    throw std::runtime_error("Restricting finite elements function in "
                             "different elements not suppoted.");
  }

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = _mesh->coordinate_dofs().entity_points(tdim);
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = _mesh->geometry().points();

  // Iterate over mesh and interpolate on each cell
  EigenRowArrayXXd coordinate_dofs(num_dofs_g, gdim);
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
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
    v.restrict(cell_coefficients.data(), cell, coordinate_dofs);

    // Tabulate dofs
    auto cell_dofs = _dofmap->cell_dofs(cell.index());

    for (Eigen::Index i = 0; i < cell_dofs.size(); ++i)
      expansion_coefficients[cell_dofs[i]] = cell_coefficients[i];
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate_from_any(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        expansion_coefficients,
    const Expression& expr) const
{
  assert(_mesh);

  const int gdim = _mesh->geometry().dim();
  const int tdim = _mesh->topology().dim();

  // Initialize local arrays
  std::vector<PetscScalar> cell_coefficients(_dofmap->max_element_dofs());

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = _mesh->coordinate_dofs().entity_points(tdim);
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = _mesh->geometry().points();

  // Iterate over mesh and interpolate on each cell
  EigenRowArrayXXd coordinate_dofs(num_dofs_g, gdim);;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Get cell coordinate dofs
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Restrict function to cell
    expr.restrict(cell_coefficients.data(), *_element, cell, coordinate_dofs);

    // Tabulate dofs
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> cell_dofs
        = _dofmap->cell_dofs(cell.index());

    // Copy dofs to vector
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
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

  // Check that function ranks match
  if (_element->value_rank() != v.value_rank())
  {
    throw std::runtime_error("Cannot interpolate function into function space. "
                             "Rank of function ("
                             + std::to_string(v.value_rank())
                             + ") does not match rank of function space ("
                             + std::to_string(element()->value_rank()) + ")");
  }

  // Check that function dimension match
  for (std::size_t i = 0; i < _element->value_rank(); ++i)
  {
    if (_element->value_dimension(i) != v.value_dimension(i))
    {
      throw std::runtime_error(
          "Cannot interpolate function into function space. "
          "Dimension "
          + std::to_string(i) + " of function ("
          + std::to_string(v.value_dimension(i)) + ") does not match dimension "
          + std::to_string(i) + " of function space("
          + std::to_string(element()->value_dimension(i)) + ")");
    }
  }

  std::shared_ptr<const FunctionSpace> v_fs = v.function_space();
  interpolate_from_any(expansion_coefficients, v);
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        expansion_coefficients,
    const Expression& expr) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

  // Check that function ranks match
  if (_element->value_rank() != expr.value_rank())
  {
    throw std::runtime_error("Rank of Expression "
                             + std::to_string(expr.value_rank())
                             + " doesn't match the target space.");
  }

  // Check that function dims match
  for (std::size_t i = 0; i < _element->value_rank(); ++i)
  {
    if (_element->value_dimension(i) != expr.value_dimension(i))
    {
      throw std::runtime_error(
          "Dimensions of Expression doesn't match the target space.");
    }
  }

  // Initialize vector of expansion coefficients
  // if (expansion_coefficients.size() != _dofmap->global_dimension())
  // {
  //   throw std::runtime_error("Cannot interpolate function into function
  //   space. "
  //                            "Wrong size of vector");
  // }
  // VecSet(expansion_coefficients.vec(), 0.0);

  interpolate_from_any(expansion_coefficients, expr);
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace>
FunctionSpace::sub(const std::vector<std::size_t>& component) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

  // Check if sub space is already in the cache and not expired
  auto subspace_it = _subspaces.find(component);
  if (subspace_it != _subspaces.end())
  {
    if (auto s = subspace_it->second.lock())
      return s;
  }

  // Extract sub-element
  std::shared_ptr<fem::FiniteElement> element
      = _element->extract_sub_element(component);

  // Extract sub dofmap
  std::shared_ptr<fem::GenericDofMap> dofmap(
      _dofmap->extract_sub_dofmap(component, *_mesh));

  // Create new sub space
  auto sub_space = std::make_shared<FunctionSpace>(_mesh, element, dofmap);

  // Set root space id and component w.r.t. root
  sub_space->_root_space_id = _root_space_id;
  sub_space->_component = _component;
  sub_space->_component.insert(sub_space->_component.end(), component.begin(),
                               component.end());

  // Insert new subspace into cache
  _subspaces.insert(
      std::pair<std::vector<std::size_t>, std::shared_ptr<FunctionSpace>>(
          sub_space->_component, sub_space));

  return sub_space;
}
//-----------------------------------------------------------------------------
std::pair<std::shared_ptr<FunctionSpace>, std::vector<PetscInt>>
FunctionSpace::collapse() const
{
  assert(_mesh);
  if (_component.empty())
  {
    throw std::runtime_error("Function space is not a subspace");
  }

  // Create collapsed DofMap
  std::shared_ptr<fem::GenericDofMap> collapsed_dofmap;
  std::vector<PetscInt> collapsed_dofs;
  std::tie(collapsed_dofmap, collapsed_dofs) = _dofmap->collapse(*_mesh);

  // Create new FunctionSpace and return
  auto collapsed_sub_space
      = std::make_shared<FunctionSpace>(_mesh, _element, collapsed_dofmap);

  return std::make_pair(std::move(collapsed_sub_space),
                        std::move(collapsed_dofs));
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> FunctionSpace::component() const { return _component; }
//-----------------------------------------------------------------------------
EigenRowArrayXXd FunctionSpace::tabulate_dof_coordinates() const
{
  // Geometric dimension
  assert(_mesh);
  assert(_element);
  const int gdim = _mesh->geometry().dim();
  const int tdim = _mesh->topology().dim();

  if (!_component.empty())
  {
    throw std::runtime_error(
        "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  // Get local size
  assert(_dofmap);
  std::size_t bs = _dofmap->index_map()->block_size();
  std::size_t local_size = bs * _dofmap->index_map()->size_local();

  // Dof coordinate on reference element
  const EigenRowArrayXXd& X = _element->dof_reference_coordinates();

  // Arrray to hold coordinates and return
  EigenRowArrayXXd x(local_size, gdim);

  // Get coordinate mapping
  if (!_mesh->geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateMapping has not been attached to mesh.");
  }
  const fem::CoordinateMapping& cmap = *_mesh->geometry().coord_mapping;

  // Cell coordinates (re-allocated inside function for thread safety)
  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = _mesh->coordinate_dofs().entity_points(tdim);
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = _mesh->geometry().points();

  // Loop over cells and tabulate dofs
  EigenRowArrayXXd coordinates(_element->space_dimension(), gdim);
  EigenRowArrayXXd coordinate_dofs(num_dofs_g, gdim);
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Update cell
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Get local-to-global map
    auto dofs = _dofmap->cell_dofs(cell.index());

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
  assert(_mesh);
  assert(_dofmap);
  assert(_element);

  const int gdim = _mesh->geometry().dim();
  const int tdim = _mesh->topology().dim();
  std::vector<PetscScalar> x_values;

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = _mesh->coordinate_dofs().entity_points(tdim);
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = _mesh->geometry().points();

  // Dof coordinate on reference element
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X
      = _element->dof_reference_coordinates();

  // Get coordinate mapping
  if (!_mesh->geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateMapping has not been attached to mesh.");
  }
  const fem::CoordinateMapping& cmap = *_mesh->geometry().coord_mapping;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinates(_element->space_dimension(), _mesh->geometry().dim());
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Update UFC cell
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Get cell local-to-global map
    auto dofs = _dofmap->cell_dofs(cell.index());

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
std::string FunctionSpace::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    // No verbose output implemented
  }
  else
    s << "<FunctionSpace of dimension " << dim() << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void FunctionSpace::print_dofmap() const
{
  assert(_mesh);
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    auto dofs = _dofmap->cell_dofs(cell.index());
    std::cout << cell.index() << ":";
    for (Eigen::Index i = 0; i < dofs.size(); i++)
      std::cout << " " << static_cast<std::size_t>(dofs[i]);
    std::cout << std::endl;
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
