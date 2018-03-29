// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(std::shared_ptr<const ufc::finite_element> element)
    : _ufc_element(element), _hash(common::hash_local(signature()))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate_dof_coordinates(
    Eigen::Ref<EigenRowArrayXXd> coordinates,
    const std::vector<double>& coordinate_dofs, const mesh::Cell& cell) const
{
  dolfin_assert(_ufc_element);

  // Check sizes
  assert((std::size_t)coordinates.rows() == this->space_dimension());
  assert((std::size_t)coordinates.cols() == this->geometric_dimension());

  // Tabulate coordinates
  _ufc_element->tabulate_dof_coordinates(coordinates.data(),
                                         coordinate_dofs.data());
}
//-----------------------------------------------------------------------------
void FiniteElement::tabulate_dof_coordinates(
    Eigen::Ref<EigenRowArrayXXd> coordinates,
    const Eigen::Ref<EigenRowArrayXXd> coordinate_dofs,
    const mesh::Cell& cell) const
{
  dolfin_assert(_ufc_element);

  // Check sizes
  assert((std::size_t)coordinates.rows() == this->space_dimension());
  assert((std::size_t)coordinates.cols() == this->geometric_dimension());

  // Tabulate coordinates
  _ufc_element->tabulate_dof_coordinates(coordinates.data(),
                                         coordinate_dofs.data());
}
//-----------------------------------------------------------------------------
std::shared_ptr<FiniteElement> FiniteElement::extract_sub_element(
    const std::vector<std::size_t>& component) const
{
  // Recursively extract sub element
  std::shared_ptr<FiniteElement> sub_finite_element
      = extract_sub_element(*this, component);
  log::log(DBG, "Extracted finite element for sub system: %s",
           sub_finite_element->signature().c_str());

  return sub_finite_element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<FiniteElement>
FiniteElement::extract_sub_element(const FiniteElement& finite_element,
                                   const std::vector<std::size_t>& component)
{
  // Check if there are any sub systems
  if (finite_element.num_sub_elements() == 0)
  {
    log::dolfin_error("FiniteElement.cpp",
                      "extract subsystem of finite element",
                      "There are no subsystems");
  }

  // Check that a sub system has been specified
  if (component.empty())
  {
    log::dolfin_error("FiniteElement.cpp",
                      "extract subsystem of finite element",
                      "No system was specified");
  }

  // Check the number of available sub systems
  if (component[0] >= finite_element.num_sub_elements())
  {
    log::dolfin_error("FiniteElement.cpp",
                      "extract subsystem of finite element",
                      "Requested subsystem (%d) out of range [0, %d)",
                      component[0], finite_element.num_sub_elements());
  }

  // Create sub system
  std::shared_ptr<FiniteElement> sub_element
      = finite_element.create_sub_element(component[0]);
  dolfin_assert(sub_element);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;

  // Otherwise, recursively extract the sub sub system
  std::vector<std::size_t> sub_component;
  for (std::size_t i = 1; i < component.size(); i++)
    sub_component.push_back(component[i]);

  return extract_sub_element(*sub_element, sub_component);
}
//-----------------------------------------------------------------------------
