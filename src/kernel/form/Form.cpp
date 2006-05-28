// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-10-21
// Last changed: 2005-11-29

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/File.h>
#include <dolfin/Form.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::Form(uint num_functions)
  : num_functions(num_functions), c(0)
{
  // Initialize list of functions
  if ( num_functions > 0 )
  {
    // Reserve list of functions
    functions.clear();
    functions.reserve(num_functions);

    // Reserve list of elements
    elements.clear();
    elements.reserve(num_functions);

    // Initialize coefficients
    c = new real* [num_functions];
    for (uint i = 0; i < num_functions; i++)
      c[i] = 0;
  }
}
//-----------------------------------------------------------------------------
Form::~Form()
{
  // Delete elements (functions are delete elsewhere)
  for (uint i = 0; i < elements.size(); i++)
    delete elements[i];

  // Delete coefficients
  if ( c )
  {
    for (uint i = 0; i < num_functions; i++)
      delete [] c[i];
    delete [] c;
  }
}
//-----------------------------------------------------------------------------
void Form::update(AffineMap& map)
{
  // Update coefficients
  updateCoefficients(map);
}
//-----------------------------------------------------------------------------
void Form::add(Function& f, FiniteElement* element)
{
  if ( functions.size() == num_functions )
    dolfin_error("All functions already added.");
  
  // Get number of new function
  uint i = functions.size();

  // Set finite element for function, but only for discrete functions
  if ( f.type() == Function::discrete )
    f.attach(*element);

  // Add function and element
  functions.push_back(&f);
  elements.push_back(element);

  // Initialize coefficients
  c[i] = new real[element->spacedim()];
  for (uint j = 0; j < element->spacedim(); j++)
    c[i][j] = 0.0;
}
//-----------------------------------------------------------------------------
void Form::updateCoefficients(AffineMap& map)
{
  dolfin_assert(num_functions == functions.size());

  // Interpolate all functions to the current element
  for (uint i = 0; i < num_functions; i++)
  {
    dolfin_assert(functions[i]);
    functions[i]->interpolate(c[i], map, *elements[i]);
  }
}
//-----------------------------------------------------------------------------
Function* Form::function(uint i)
{
  return functions[i];
}
//-----------------------------------------------------------------------------
FiniteElement* Form::element(uint i)
{
  return elements[i];
}
//-----------------------------------------------------------------------------
