// Copyright (C) 2008-2009 Anders Logg
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
// Modified by Kristoffer Selim, 2008.
// Modified by Martin Alnes, 2008.
// Modified by Garth N. Wells, 2008-2011.
// Modified by Kent-Andre Mardal, 2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2008-09-11
// Last changed: 2015-11-12

#include <vector>
#include <dolfin/common/utils.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include "GenericFunction.h"
#include "FunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::shared_ptr<const Mesh> mesh,
                             std::shared_ptr<const FiniteElement> element,
                             std::shared_ptr<const GenericDofMap> dofmap)
  : Hierarchical<FunctionSpace>(*this),
    _mesh(mesh), _element(element), _dofmap(dofmap)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(std::shared_ptr<const Mesh> mesh)
  : Hierarchical<FunctionSpace>(*this), _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(const FunctionSpace& V)
  : Hierarchical<FunctionSpace>(*this)
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
void FunctionSpace::attach(std::shared_ptr<const FiniteElement> element,
                           std::shared_ptr<const GenericDofMap> dofmap)
{
  _element = element;
  _dofmap  = dofmap;
}
//-----------------------------------------------------------------------------
const FunctionSpace& FunctionSpace::operator=(const FunctionSpace& V)
{
  // Assign data (will be shared)
  _mesh      = V._mesh;
  _element   = V._element;
  _dofmap    = V._dofmap;
  _component = V._component;

  // Call assignment operator for base class
  Hierarchical<FunctionSpace>::operator=(V);

  return *this;
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator==(const FunctionSpace& V) const
{
  // Compare pointers to shared objects
  return _element.get() == V._element.get() &&
    _mesh.get() == V._mesh.get() &&
    _dofmap.get() == V._dofmap.get();
}
//-----------------------------------------------------------------------------
bool FunctionSpace::operator!=(const FunctionSpace& V) const
{
  // Compare pointers to shared objects
  return !(*this == V);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Mesh> FunctionSpace::mesh() const
{
  return _mesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement> FunctionSpace::element() const
{
  return _element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericDofMap> FunctionSpace::dofmap() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
std::size_t FunctionSpace::dim() const
{
  dolfin_assert(_dofmap);
  return _dofmap->global_dimension();
}
//-----------------------------------------------------------------------------
void
FunctionSpace::interpolate_from_parent(GenericVector& expansion_coefficients,
                                       const GenericFunction& v) const
{
  info("Interpolate from parent to child");

  std::shared_ptr<const FunctionSpace> v_fs = v.function_space();

  // Initialize local arrays
  std::vector<double> cell_coefficients(_dofmap->max_element_dofs());

  // Iterate over mesh and interpolate on each cell
  std::vector<double> coordinate_dofs;
  std::size_t tdim = _mesh->topology().dim();
  const std::vector<std::size_t>& child_to_parent = _mesh->data().array("parent_cell", tdim);

  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    cell->get_coordinate_dofs(coordinate_dofs);

    // Get cell orientation
    int cell_orientation = -1;
    if (!_mesh->cell_orientations().empty())
    {
      dolfin_assert(cell->index() < _mesh->cell_orientations().size());
      cell_orientation = _mesh->cell_orientations()[cell->index()];
    }

    Cell parent_cell(*v_fs->mesh(), child_to_parent[cell->index()]);
    ufc::cell ufc_parent;
    parent_cell.get_cell_data(ufc_parent);

    // Evaluate on parent cell on which v is defined
    _element->evaluate_dofs(cell_coefficients.data(), v,
                            coordinate_dofs.data(),
                            cell_orientation,
                            ufc_parent);

    // Tabulate dofs - map from cell to vector
    const ArrayView<const dolfin::la_index> cell_dofs
      = _dofmap->cell_dofs(cell->index());

    // Copy dofs to vector
    expansion_coefficients.set_local(cell_coefficients.data(),
                                     _dofmap->num_element_dofs(cell->index()),
                                     cell_dofs.data());
  }

}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate_from_any(GenericVector& expansion_coefficients,
                                         const GenericFunction& v) const
{
  // Initialize local arrays
  std::vector<double> cell_coefficients(_dofmap->max_element_dofs());

  // Iterate over mesh and interpolate on each cell
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    cell->get_coordinate_dofs(coordinate_dofs);
    cell->get_cell_data(ufc_cell);

    // Restrict function to cell
    v.restrict(cell_coefficients.data(), *_element, *cell,
               coordinate_dofs.data(), ufc_cell);

    // Tabulate dofs
    const ArrayView<const dolfin::la_index> cell_dofs
      = _dofmap->cell_dofs(cell->index());

    // Copy dofs to vector
    expansion_coefficients.set_local(cell_coefficients.data(),
                                     _dofmap->num_element_dofs(cell->index()),
                                     cell_dofs.data());
  }

}
//-----------------------------------------------------------------------------
void FunctionSpace::interpolate(GenericVector& expansion_coefficients,
                                const GenericFunction& v) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Check that function ranks match
  if (_element->value_rank() != v.value_rank())
  {
    dolfin_error("FunctionSpace.cpp",
                 "interpolate function into function space",
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
                   "Dimension %d of function (%d) does not match dimension %d of function space (%d)",
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

  // Interpolate from parent to child
  // should also work in parallel provided "parent_cell" data exists
  if (v_fs and _mesh->has_parent()
      and v_fs->mesh()->id() == _mesh->parent().id()
      and _mesh->data().exists("parent_cell", _mesh->topology().dim()))
  {
    interpolate_from_parent(expansion_coefficients, v);
  }
  else
    interpolate_from_any(expansion_coefficients, v);

  // Finalise changes
  expansion_coefficients.apply("insert");
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace> FunctionSpace::operator[] (std::size_t i) const
{
  std::vector<std::size_t> component;
  component.push_back(i);
  return extract_sub_space(component);
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace>
FunctionSpace::extract_sub_space(const std::vector<std::size_t>& component) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Check if sub space is already in the cache
  std::map<std::vector<std::size_t>,
           std::shared_ptr<FunctionSpace>>::const_iterator subspace;
  subspace = _subspaces.find(component);
  if (subspace != _subspaces.end())
    return subspace->second;
  else
  {
    // Extract sub element
    std::shared_ptr<const FiniteElement>
      element(_element->extract_sub_element(component));

    // Extract sub dofmap
    std::shared_ptr<GenericDofMap>
      dofmap(_dofmap->extract_sub_dofmap(component, *_mesh));

    // Create new sub space
    std::shared_ptr<FunctionSpace>
      new_sub_space(new FunctionSpace(_mesh, element, dofmap));

    // Set component
    new_sub_space->_component.resize(component.size());
    for (std::size_t i = 0; i < component.size(); i++)
      new_sub_space->_component[i] = component[i];

    // Insert new sub space into cache
    _subspaces.insert(std::pair<std::vector<std::size_t>,
                      std::shared_ptr<FunctionSpace>>(component,
                                                      new_sub_space));

    return new_sub_space;
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace> FunctionSpace::collapse() const
{
  std::unordered_map<std::size_t, std::size_t> collapsed_dofs;
  return collapse(collapsed_dofs);
}
//-----------------------------------------------------------------------------
std::shared_ptr<FunctionSpace>FunctionSpace::collapse(
  std::unordered_map<std::size_t, std::size_t>& collapsed_dofs) const
{
  dolfin_assert(_mesh);

  if (_component.empty())
  {
    dolfin_error("FunctionSpace.cpp",
                 "collapse function space",
                 "Function space is not a subspace");
  }

  // Create collapsed DofMap
  std::shared_ptr<GenericDofMap>
    collapsed_dofmap(_dofmap->collapse(collapsed_dofs, *_mesh));

  // Create new FunctionSpace and return
  std::shared_ptr<FunctionSpace>
    collapsed_sub_space(new FunctionSpace(_mesh, _element, collapsed_dofmap));
  return collapsed_sub_space;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> FunctionSpace::component() const
{
  return _component;
}
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
    dolfin_error("FunctionSpace.cpp",
                 "tabulate_dof_coordinates",
                 "Cannot tabulate coordinates for a FunctionSpace that is a subspace.");
  }

  // Get local size
  dolfin_assert(_dofmap);
  std::size_t local_size
    = _dofmap->index_map()->size(IndexMap::MapSize::OWNED);

  // Vector to hold coordinates and return
  std::vector<double> x(gdim*local_size);

  // Loop over cells and tabulate dofs
  boost::multi_array<double, 2> coordinates;
  std::vector<double> coordinate_dofs;
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    cell->get_coordinate_dofs(coordinate_dofs);

    // Get local-to-global map
    const ArrayView<const dolfin::la_index> dofs
      = _dofmap->cell_dofs(cell->index());

    // Tabulate dof coordinates on cell
    _element->tabulate_dof_coordinates(coordinates, coordinate_dofs, *cell);

    // Copy dof coordinates into vector
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      const dolfin::la_index dof = dofs[i];
      if (dof < (dolfin::la_index) local_size)
      {
        const dolfin::la_index local_index = dof;
        for (std::size_t j = 0; j < gdim; ++j)
        {
          dolfin_assert(gdim*local_index + j < x.size());
          x[gdim*local_index + j] = coordinates[i][j];
        }
      }
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
void FunctionSpace::set_x(GenericVector& x, double value,
                          std::size_t component) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_dofmap);
  dolfin_assert(_element);

  std::vector<double> x_values;
  boost::multi_array<double, 2> coordinates;
  std::vector<double> coordinate_dofs;
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    cell->get_coordinate_dofs(coordinate_dofs);

    // Get cell local-to-global map
    const ArrayView<const dolfin::la_index> dofs
      = _dofmap->cell_dofs(cell->index());

    // Tabulate dof coordinates
    _element->tabulate_dof_coordinates(coordinates, coordinate_dofs, *cell);
    dolfin_assert(coordinates.shape()[0] == dofs.size());
    dolfin_assert(component < coordinates.shape()[1]);

    // Copy coordinate (it may be possible to avoid this)
    x_values.resize(dofs.size());
    for (std::size_t i = 0; i < coordinates.shape()[0]; ++i)
      x_values[i] = value*coordinates[i][component];

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
  // Note: static_cast is used below to support types that cannot be
  //       directed to dolfin::cout
  dolfin_assert(_mesh);
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    const ArrayView<const dolfin::la_index> dofs
      = _dofmap->cell_dofs(cell->index());
    cout << cell->index() << ":";
    for (std::size_t i = 0; i < dofs.size(); i++)
      cout << " " << static_cast<std::size_t>(dofs[i]);
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
