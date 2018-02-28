// Copyright (C) 2008-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FunctionSpace.h"
#include "Function.h"
#include "GenericFunction.h"
#include <dolfin/common/utils.h>
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
  dolfin_assert(_dofmap);
  return _dofmap->global_dimension();
}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate_from_any(
    la::PETScVector& expansion_coefficients, const GenericFunction& v) const
{
  // Initialize local arrays
  std::vector<double> cell_coefficients(_dofmap->max_element_dofs());

  // Iterate over mesh and interpolate on each cell
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Update to current cell
    cell.get_coordinate_dofs(coordinate_dofs);
    cell.get_cell_data(ufc_cell);

    // Restrict function to cell
    v.restrict(cell_coefficients.data(), *_element, cell,
               coordinate_dofs.data(), ufc_cell);

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
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Check that function ranks match
  if (_element->value_rank() != v.value_rank())
  {
    dolfin_error(
        "FunctionSpace.cpp", "interpolate function into function space",
        "Rank of function (%d) does not match rank of function space (%d)",
        v.value_rank(), element()->value_rank());
  }

  // Check that function dims match
  for (std::size_t i = 0; i < _element->value_rank(); ++i)
  {
    if (_element->value_dimension(i) != v.value_dimension(i))
    {
      dolfin_error("FunctionSpace.cpp",
                   "interpolate function into function space",
                   "Dimension %d of function (%d) does not match dimension %d "
                   "of function space (%d)",
                   i, v.value_dimension(i), i, element()->value_dimension(i));
    }
  }

  // Initialize vector of expansion coefficients
  if (expansion_coefficients.size() != _dofmap->global_dimension())
  {
    dolfin_error("FunctionSpace.cpp",
                 "interpolate function into function space",
                 "Wrong size of vector");
  }
  expansion_coefficients.zero();

  std::shared_ptr<const FunctionSpace> v_fs = v.function_space();

  interpolate_from_any(expansion_coefficients, v);

  // Finalise changes
  expansion_coefficients.apply();
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace>
FunctionSpace::sub(const std::vector<std::size_t>& component) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

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
std::shared_ptr<FunctionSpace> FunctionSpace::collapse() const
{
  std::unordered_map<std::size_t, std::size_t> collapsed_dofs;
  return collapse(collapsed_dofs);
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace> FunctionSpace::collapse(
    std::unordered_map<std::size_t, std::size_t>& collapsed_dofs) const
{
  dolfin_assert(_mesh);

  if (_component.empty())
  {
    dolfin_error("FunctionSpace.cpp", "collapse function space",
                 "Function space is not a subspace");
  }

  // Create collapsed DofMap
  std::shared_ptr<fem::GenericDofMap> collapsed_dofmap(
      _dofmap->collapse(collapsed_dofs, *_mesh));

  // Create new FunctionSpace and return
  std::shared_ptr<FunctionSpace> collapsed_sub_space(
      new FunctionSpace(_mesh, _element, collapsed_dofmap));
  return collapsed_sub_space;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> FunctionSpace::component() const { return _component; }
//-----------------------------------------------------------------------------
std::vector<double> FunctionSpace::tabulate_dof_coordinates() const
{
  // Geometric dimension
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  const std::size_t gdim = _element->geometric_dimension();
  dolfin_assert(gdim == _mesh->geometry().dim());

  if (!_component.empty())
  {
    dolfin_error(
        "FunctionSpace.cpp", "tabulate_dof_coordinates",
        "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  // Get local size
  dolfin_assert(_dofmap);
  std::size_t bs = _dofmap->block_size();
  std::size_t local_size
      = bs * _dofmap->index_map()->size(common::IndexMap::MapSize::OWNED);

  // Vector to hold coordinates and return
  std::vector<double> x(gdim * local_size);

  // Loop over cells and tabulate dofs
  boost::multi_array<double, 2> coordinates;
  std::vector<double> coordinate_dofs;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Update UFC cell
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get local-to-global map
    auto dofs = _dofmap->cell_dofs(cell.index());

    // Tabulate dof coordinates on cell
    _element->tabulate_dof_coordinates(coordinates, coordinate_dofs, cell);

    // Copy dof coordinates into vector
    for (Eigen::Index i = 0; i < dofs.size(); ++i)
    {
      const dolfin::la_index_t dof = dofs[i];
      if (dof < (dolfin::la_index_t)local_size)
      {
        const dolfin::la_index_t local_index = dof;
        for (std::size_t j = 0; j < gdim; ++j)
        {
          dolfin_assert(gdim * local_index + j < x.size());
          x[gdim * local_index + j] = coordinates[i][j];
        }
      }
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
void FunctionSpace::set_x(la::PETScVector& x, double value,
                          std::size_t component) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_dofmap);
  dolfin_assert(_element);

  std::vector<double> x_values;
  boost::multi_array<double, 2> coordinates;
  std::vector<double> coordinate_dofs;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(*_mesh))
  {
    // Update UFC cell
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get cell local-to-global map
    auto dofs = _dofmap->cell_dofs(cell.index());

    // Tabulate dof coordinates
    _element->tabulate_dof_coordinates(coordinates, coordinate_dofs, cell);
    dolfin_assert(coordinates.shape()[0] == (std::size_t)dofs.size());
    dolfin_assert(component < coordinates.shape()[1]);

    // Copy coordinate (it may be possible to avoid this)
    x_values.resize(dofs.size());
    for (std::size_t i = 0; i < coordinates.shape()[0]; ++i)
      x_values[i] = value * coordinates[i][component];

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
  dolfin_assert(_mesh);
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
