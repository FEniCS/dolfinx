// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-14
// Last changed: 2008-10-14

#include <dolfin/log/log.h>
#include "FiniteElement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FiniteElement* FiniteElement::extract_sub_element(const Array<uint>& sub_system) const
{
  // Recursively extract sub element
  FiniteElement* sub_finite_element = extract_sub_element(*this, sub_system);
  message(2, "Extracted finite element for sub system: %s", sub_finite_element->signature().c_str());
  
  return sub_finite_element;
}
//-----------------------------------------------------------------------------
FiniteElement* FiniteElement::extract_sub_element(const FiniteElement& finite_element,
                                                  const Array<uint>& sub_system)
{
  // Check if there are any sub systems
  if (finite_element.num_sub_elements() == 0)
    error("Unable to extract sub system (there are no sub systems).");

  // Check that a sub system has been specified
  if (sub_system.size() == 0)
    error("Unable to extract sub system (no sub system specified).");
  
  // Check the number of available sub systems
  if (sub_system[0] >= finite_element.num_sub_elements())
    error("Unable to extract sub system %d (only %d sub systems defined).",
          sub_system[0], finite_element.num_sub_elements());
  
  // Create sub system
  FiniteElement* sub_element = finite_element.create_sub_element(sub_system[0]);
  
  // Return sub system if sub sub system should not be extracted
  if (sub_system.size() == 1)
    return sub_element;
  
  // Otherwise, recursively extract the sub sub system
  Array<uint> sub_sub_system;
  for (uint i = 1; i < sub_system.size(); i++)
    sub_sub_system.push_back(sub_system[i]);
  FiniteElement* sub_sub_element = extract_sub_element(*sub_element, sub_sub_system);
  delete sub_element;
  
  return sub_sub_element;
}
//-----------------------------------------------------------------------------
