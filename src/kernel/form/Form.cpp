// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-10-21
// Last changed: 2005-09-29

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/File.h>
#include <dolfin/Form.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::Form(uint num_functions)
  : c(0), num_functions(num_functions), blas_A(0), blas_G(0)
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

  // Delete form data for BLAS
  delete [] blas_A;
  delete [] blas_G;
}
//-----------------------------------------------------------------------------
void Form::update(const AffineMap& map)
{
  // Update coefficients
  updateCoefficients(map);
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

  // Initialize coefficients
  c[i] = new real[element->spacedim()];
  for (uint j = 0; j < element->spacedim(); j++)
    c[i][j] = 0.0;
}
//-----------------------------------------------------------------------------
void Form::updateCoefficients(const AffineMap& map)
{
  dolfin_assert(num_functions == functions.size());

  // Interpolate all functions to the current element
  for (uint i = 0; i < num_functions; i++)
  {
    dolfin_assert(functions[i]);
    functions[i]->interpolate(c[i], map);
  }
}
//-----------------------------------------------------------------------------
void initBLAS(const char* filename)
{
  //File file(filename);
  //file << *this;
}
//-----------------------------------------------------------------------------
