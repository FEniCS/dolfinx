// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-14
// Last changed: 2008-11-03

#include <dolfin/log/log.h>
#include "FiniteElement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::tr1::shared_ptr<const FiniteElement> FiniteElement::extract_sub_element(const std::vector<uint>& component) const
{
  // Recursively extract sub element
  std::tr1::shared_ptr<const FiniteElement> sub_finite_element = extract_sub_element(*this, component);
  message(2, "Extracted finite element for sub system: %s", sub_finite_element->signature().c_str());
  
  return sub_finite_element;
}
//-----------------------------------------------------------------------------
std::tr1::shared_ptr<const FiniteElement> FiniteElement::extract_sub_element(const FiniteElement& finite_element, 
                                      const std::vector<uint>& component)
{
  // Check if there are any sub systems
  if (finite_element.num_sub_elements() == 0)
    error("Unable to extract sub system (there are no sub systems).");

  // Check that a sub system has been specified
  if (component.size() == 0)
    error("Unable to extract sub system (no sub system specified).");
  
  // Check the number of available sub systems
  if (component[0] >= finite_element.num_sub_elements())
    error("Unable to extract sub system %d (only %d sub systems defined).",
          component[0], finite_element.num_sub_elements());
  
  // Create sub system
  std::tr1::shared_ptr<const FiniteElement> sub_element = finite_element.create_sub_element(component[0]);
  
  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;
  
  // Otherwise, recursively extract the sub sub system
  std::vector<uint> sub_component;
  for (uint i = 1; i < component.size(); i++)
    sub_component.push_back(component[i]);
  std::tr1::shared_ptr<const FiniteElement> sub_sub_element = extract_sub_element(*sub_element, sub_component);
  //delete sub_element;
  
  return sub_sub_element;
}
//-----------------------------------------------------------------------------
