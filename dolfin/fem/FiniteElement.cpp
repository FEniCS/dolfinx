// Copyright (C) 2008-2011 Anders Logg
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
// First added:  2008-10-14
// Last changed: 2011-11-14

#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include "FiniteElement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(std::shared_ptr<const ufc::finite_element> element)
  : _ufc_element(element), _hash(dolfin::hash_local(signature()))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate_dof_coordinates(
  boost::multi_array<double, 2>& coordinates,
  const std::vector<double>& coordinate_dofs,
  const Cell& cell) const
{
  dolfin_assert(_ufc_element);

  // Check dimensions
  const std::size_t dim = this->space_dimension();
  const std::size_t gdim = this->geometric_dimension();
  if (coordinates.shape()[0] != dim or coordinates.shape()[1] != gdim)
  {
    boost::multi_array<double, 2>::extent_gen extents;
    coordinates.resize(extents[dim][gdim]);
  }

  // Tabulate coordinates
  _ufc_element->tabulate_dof_coordinates(coordinates.data(),
                                         coordinate_dofs.data());
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement>
FiniteElement::extract_sub_element(const std::vector<std::size_t>& component) const
{
  // Recursively extract sub element
  std::shared_ptr<const FiniteElement> sub_finite_element
    = extract_sub_element(*this, component);
  log(DBG, "Extracted finite element for sub system: %s",
      sub_finite_element->signature().c_str());

  return sub_finite_element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement>
FiniteElement::extract_sub_element(const FiniteElement& finite_element,
                                   const std::vector<std::size_t>& component)
{
  // Check if there are any sub systems
  if (finite_element.num_sub_elements() == 0)
  {
    dolfin_error("FiniteElement.cpp",
                 "extract subsystem of finite element",
                 "There are no subsystems");
  }

  // Check that a sub system has been specified
  if (component.empty())
  {
    dolfin_error("FiniteElement.cpp",
                 "extract subsystem of finite element",
                 "No system was specified");
  }

  // Check the number of available sub systems
  if (component[0] >= finite_element.num_sub_elements())
  {
    dolfin_error("FiniteElement.cpp",
                 "extract subsystem of finite element",
                 "Requested subsystem (%d) out of range [0, %d)",
                 component[0], finite_element.num_sub_elements());
  }

  // Create sub system
  std::shared_ptr<const FiniteElement> sub_element
    = finite_element.create_sub_element(component[0]);
  dolfin_assert(sub_element);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;

  // Otherwise, recursively extract the sub sub system
  std::vector<std::size_t> sub_component;
  for (std::size_t i = 1; i < component.size(); i++)
    sub_component.push_back(component[i]);

  std::shared_ptr<const FiniteElement> sub_sub_element
    = extract_sub_element(*sub_element, sub_component);

  return sub_sub_element;
}
//-----------------------------------------------------------------------------
