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
// Last changed: 2011-05-15

#include <dolfin/common/utils.h>
#include <dolfin/common/MPI.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/UFCCell.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include "GenericFunction.h"
#include "Function.h"
#include "FunctionSpace.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(boost::shared_ptr<const Mesh> mesh,
                             boost::shared_ptr<const FiniteElement> element,
                             boost::shared_ptr<const GenericDofMap> dofmap)
  : Hierarchical<FunctionSpace>(*this),
    _mesh(mesh), _element(element), _dofmap(dofmap)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace::FunctionSpace(boost::shared_ptr<const Mesh> mesh)
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
void FunctionSpace::attach(boost::shared_ptr<const FiniteElement> element,
                           boost::shared_ptr<const GenericDofMap> dofmap)
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
boost::shared_ptr<const Mesh> FunctionSpace::mesh() const
{
  return _mesh;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const FiniteElement> FunctionSpace::element() const
{
  return _element;
}
//-----------------------------------------------------------------------------
boost::shared_ptr<const GenericDofMap> FunctionSpace::dofmap() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
dolfin::uint FunctionSpace::dim() const
{
  dolfin_assert(_dofmap);
  return _dofmap->global_dimension();
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
  for (uint i = 0; i < _element->value_rank(); ++i)
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
  //expansion_coefficients.resize(_dofmap->global_dimension());
  if (expansion_coefficients.size() != _dofmap->global_dimension())
  {
    dolfin_error("FunctionSpace.cpp",
                 "interpolate function into function space",
                 "Wrong size of vector");
  }
  expansion_coefficients.zero();

  // Initialize local arrays
  std::vector<double> cell_coefficients(_dofmap->max_cell_dimension());

  // Iterate over mesh and interpolate on each cell
  UFCCell ufc_cell(*_mesh);
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    ufc_cell.update(*cell);

    // Restrict function to cell
    v.restrict(&cell_coefficients[0], *_element, *cell, ufc_cell);

    // Tabulate dofs
    const std::vector<uint>& cell_dofs = _dofmap->cell_dofs(cell->index());

    // Copy dofs to vector
    expansion_coefficients.set(&cell_coefficients[0],
                               _dofmap->cell_dimension(cell->index()),
                               &cell_dofs[0]);
  }

  // Finalise changes
  expansion_coefficients.apply("insert");
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace> FunctionSpace::operator[] (uint i) const
{
  std::vector<uint> component;
  component.push_back(i);
  return extract_sub_space(component);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace>
FunctionSpace::extract_sub_space(const std::vector<uint>& component) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_element);
  dolfin_assert(_dofmap);

  // Check if sub space is already in the cache
  std::map<std::vector<uint>, boost::shared_ptr<FunctionSpace> >::const_iterator subspace;
  subspace = subspaces.find(component);
  if (subspace != subspaces.end())
    return subspace->second;
  else
  {
    // Extract sub element
    boost::shared_ptr<const FiniteElement> element(_element->extract_sub_element(component));

    // Extract sub dofmap
    boost::shared_ptr<GenericDofMap> dofmap(_dofmap->extract_sub_dofmap(component, *_mesh));

    // Create new sub space
    boost::shared_ptr<FunctionSpace> new_sub_space(new FunctionSpace(_mesh, element, dofmap));

    // Set component
    new_sub_space->_component.resize(component.size());
    for (uint i = 0; i < component.size(); i++)
      new_sub_space->_component[i] = component[i];

    // Insert new sub space into cache
    subspaces.insert(std::pair<std::vector<uint>, boost::shared_ptr<FunctionSpace> >(component, new_sub_space));

    return new_sub_space;
  }
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace> FunctionSpace::collapse() const
{
  boost::unordered_map<uint, uint> collapsed_dofs;
  return collapse(collapsed_dofs);
}
//-----------------------------------------------------------------------------
boost::shared_ptr<FunctionSpace>
FunctionSpace::collapse(boost::unordered_map<uint, uint>& collapsed_dofs) const
{
  dolfin_assert(_mesh);

  if (_component.empty())
  {
    dolfin_error("FunctionSpace.cpp",
                 "collapse function space",
                 "Function space is not a subspace");
  }

  // Create collapsed DofMap
  boost::shared_ptr<GenericDofMap> collapsed_dofmap(_dofmap->collapse(collapsed_dofs, *_mesh));

  // Create new FunctionsSpace and return
  boost::shared_ptr<FunctionSpace> collapsed_sub_space(new FunctionSpace(_mesh, _element, collapsed_dofmap));
  return collapsed_sub_space;
}
//-----------------------------------------------------------------------------
std::vector<dolfin::uint> FunctionSpace::component() const
{
  return _component;
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
  for (CellIterator cell(*_mesh); !cell.end(); ++cell)
  {
    const std::vector<uint>& dofs = _dofmap->cell_dofs(cell->index());

    cout << cell->index() << ":";
    for (uint i = 0; i < dofs.size(); i++)
      cout << " " << dofs[i];
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
