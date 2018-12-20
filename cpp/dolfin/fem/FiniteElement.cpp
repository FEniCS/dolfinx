// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include <memory>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(std::shared_ptr<const ufc_finite_element> element)
    : _ufc_element(element), _hash(common::hash_local(signature()))
{
  // Store dof coordinates on reference element
  assert(_ufc_element);
  _refX.resize(this->space_dimension(), this->topological_dimension());
  int ret = _ufc_element->tabulate_reference_dof_coordinates(_refX.data());
  if (ret == -1)
    throw std::runtime_error("Generated code returned error "
                             "in tabulate_reference_dof_coordinates");
}
//-----------------------------------------------------------------------------
std::unique_ptr<FiniteElement>
FiniteElement::create_sub_element(std::size_t i) const
{
  assert(_ufc_element);
  std::shared_ptr<ufc_finite_element> ufc_element(
      _ufc_element->create_sub_element(i));
  return std::make_unique<FiniteElement>(ufc_element);
}
//-----------------------------------------------------------------------------
std::unique_ptr<FiniteElement> FiniteElement::create() const
{
  assert(_ufc_element);
  std::shared_ptr<ufc_finite_element> ufc_element(_ufc_element->create());
  return std::make_unique<FiniteElement>(ufc_element);
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
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. There are no subsystems.");
  }

  // Check that a sub system has been specified
  if (component.empty())
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. No system was specified");
  }

  // Check the number of available sub systems
  if (component[0] >= finite_element.num_sub_elements())
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. Requested "
        "subsystem out of range.");
  }

  // Create sub system
  std::shared_ptr<FiniteElement> sub_element
      = finite_element.create_sub_element(component[0]);
  assert(sub_element);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;

  // Otherwise, recursively extract the sub sub system
  const std::vector<std::size_t> sub_component(component.begin() + 1,
                                               component.end());

  return extract_sub_element(*sub_element, sub_component);
}
//-----------------------------------------------------------------------------
