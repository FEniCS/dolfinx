// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Cell.h>
#include <dolfin/Point.h>
#include <dolfin/Function.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Form.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::Form(uint num_functions) : w(0), num_functions(num_functions)
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
    w = new real* [num_functions];
    for (uint i = 0; i < num_functions; i++)
      w[i] = 0;
  }
}
//-----------------------------------------------------------------------------
Form::~Form()
{
  // Delete elements (functions are delete elsewhere)
  for (uint i = 0; i < elements.size(); i++)
    delete elements[i];

  // Delete coefficients
  if ( w )
  {
    for (uint i = 0; i < num_functions; i++)
      delete [] w[i];
    delete [] w;
  }
}
//-----------------------------------------------------------------------------
void Form::update(const Cell& cell)
{
  // Update coefficients
  updateCoefficients(cell);
}
//-----------------------------------------------------------------------------
void Form::updateCoefficients(const Cell& cell)
{
  dolfin_assert(num_functions == functions.size());

  // Compute the projection of all functions to the current element
  for (uint i = 0; i < num_functions; i++)
  {
    dolfin_assert(functions[i]);
    functions[i]->project(cell, w[i]);
  }
}
//-----------------------------------------------------------------------------
void Form::add(Function& function, const FiniteElement* element)
{
  if ( functions.size() == num_functions )
    dolfin_error("All functions already added.");
  
  // Get number of new function
  uint i = functions.size();

  // Set finite element for function
  function.set(*element);

  // Add function and element
  functions.push_back(&function);
  elements.push_back(element);

  // Initialize coefficient
  w[i] = new real[element->spacedim()];
  for (uint j = 0; j < element->spacedim(); j++)
    w[i][j] = 0.0;
}
//-----------------------------------------------------------------------------
