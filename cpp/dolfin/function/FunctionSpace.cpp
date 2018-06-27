// Copyright (C) 2008-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FunctionSpace.h"
#include "Function.h"
#include "GenericFunction.h"
#include <dolfin/common/types.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/log/log.h>
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
    la::PETScVector& expansion_coefficients, const GenericFunction& v) const
{
  assert(_mesh);

  std::size_t gdim = _mesh->geometry().dim();

  // Initialize local arrays
  std::vector<PetscScalar> cell_coefficients(_dofmap->max_element_dofs());

  // Iterate over mesh and interpolate on each cell
  EigenRowArrayXXd coordinate_dofs;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Get cell coordinate dofs
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);

    // Restrict function to cell
    v.restrict(cell_coefficients.data(), *_element, cell, coordinate_dofs);

    // Tabulate dofs
    auto cell_dofs = _dofmap->cell_dofs(cell.index());

    // Copy dofs to vector
    expansion_coefficients.set_local(cell_coefficients.data(),
                                     _dofmap->num_element_dofs(cell.index()),
                                     cell_dofs.data());
  }
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(la::PETScVector& expansion_coefficients,
                                const GenericFunction& v) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

  // Check that function ranks match
  if (_element->value_rank() != v.value_rank())
  {
    log::dolfin_error(
        "FunctionSpace.cpp", "interpolate function into function space",
        "Rank of function (%d) does not match rank of function space (%d)",
        v.value_rank(), element()->value_rank());
  }

  // Check that function dims match
  for (std::size_t i = 0; i < _element->value_rank(); ++i)
  {
    if (_element->value_dimension(i) != v.value_dimension(i))
    {
      log::dolfin_error(
          "FunctionSpace.cpp", "interpolate function into function space",
          "Dimension %d of function (%d) does not match dimension %d "
          "of function space (%d)",
          i, v.value_dimension(i), i, element()->value_dimension(i));
    }
  }

  // Initialize vector of expansion coefficients
  if (expansion_coefficients.size() != _dofmap->global_dimension())
  {
    throw std::runtime_error("Cannot interpolate function into function space. "
                             "Wrong size of vector");
  }
  expansion_coefficients.set(0.0);

  std::shared_ptr<const FunctionSpace> v_fs = v.function_space();
  interpolate_from_any(expansion_coefficients, v);

  // Finalise changes
  expansion_coefficients.apply();
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace>
FunctionSpace::sub(const std::vector<std::size_t>& component) const
{
  assert(_mesh);
  assert(_element);
  assert(_dofmap);

  // Check if sub space is already in the cache and not expired
  auto subspace = _subspaces.find(component);
  if (subspace != _subspaces.end())
    if (auto s = subspace->second.lock())
      return s;

  // Extract sub-element
  auto element = _element->extract_sub_element(component);

  // Extract sub dofmap
  std::shared_ptr<fem::GenericDofMap> dofmap(
      _dofmap->extract_sub_dofmap(component, *_mesh));

  // Create new sub space
  auto new_sub_space = std::make_shared<FunctionSpace>(_mesh, element, dofmap);

  // Set root space id and component w.r.t. root
  new_sub_space->_root_space_id = _root_space_id;
  auto& new_component = new_sub_space->_component;
  new_component.clear();
  new_component.insert(new_component.end(), _component.begin(),
                       _component.end());
  new_component.insert(new_component.end(), component.begin(), component.end());

  // Insert new subspace into cache
  _subspaces.insert(
      std::pair<std::vector<std::size_t>, std::shared_ptr<FunctionSpace>>(
          component, new_sub_space));

  return new_sub_space;
}
//-----------------------------------------------------------------------------
std::pair<std::shared_ptr<FunctionSpace>,
          std::unordered_map<std::size_t, std::size_t>>
FunctionSpace::collapse() const
{
  assert(_mesh);
  if (_component.empty())
  {
    log::dolfin_error("FunctionSpace.cpp", "collapse function space",
                      "Function space is not a subspace");
  }

  // Create collapsed DofMap
  std::shared_ptr<fem::GenericDofMap> collapsed_dofmap;
  std::unordered_map<std::size_t, std::size_t> collapsed_dofs;
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
  const std::size_t gdim = _mesh->geometry().dim();

  if (!_component.empty())
  {
    log::dolfin_error(
        "FunctionSpace.cpp", "tabulate_dof_coordinates",
        "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  // Get local size
  assert(_dofmap);
  std::size_t bs = _dofmap->block_size();
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

  // Loop over cells and tabulate dofs
  EigenRowArrayXXd coordinates(_element->space_dimension(), gdim);
  EigenRowArrayXXd coordinate_dofs;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Update UFC cell
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get local-to-global map
    auto dofs = _dofmap->cell_dofs(cell.index());

    // Tabulate dof coordinates on cell
    cmap.compute_physical_coordinates(coordinates, X, coordinate_dofs);

    // Copy dof coordinates into vector
    for (Eigen::Index i = 0; i < dofs.size(); ++i)
    {
      const dolfin::la_index_t dof = dofs[i];
      if (dof < (dolfin::la_index_t)local_size)
        x.row(dof) = coordinates.row(i);
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
void FunctionSpace::set_x(la::PETScVector& x, PetscScalar value,
                          std::size_t component) const
{
  assert(_mesh);
  assert(_dofmap);
  assert(_element);

  const std::size_t gdim = _mesh->geometry().dim();
  std::vector<PetscScalar> x_values;

  // Dof coordinate on reference element
  const EigenRowArrayXXd& X = _element->dof_reference_coordinates();

  // Get coordinate mapping
  if (!_mesh->geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateMapping has not been attached to mesh.");
  }
  const fem::CoordinateMapping& cmap = *_mesh->geometry().coord_mapping;

  EigenRowArrayXXd coordinates(_element->space_dimension(),
                               _mesh->geometry().dim());
  EigenRowArrayXXd coordinate_dofs;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Update UFC cell
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get cell local-to-global map
    auto dofs = _dofmap->cell_dofs(cell.index());

    // Tabulate dof coordinates
    cmap.compute_physical_coordinates(coordinates, X, coordinate_dofs);

    assert(coordinates.rows() == dofs.size());
    assert(component < (std::size_t)coordinates.cols());

    // Copy coordinate (it may be possible to avoid this)
    x_values.resize(dofs.size());
    for (Eigen::Index i = 0; i < coordinates.rows(); ++i)
      x_values[i] = value * coordinates(i, component);

    // Set x[component] values in vector
    x.set_local(x_values.data(), dofs.size(), dofs.data());
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
