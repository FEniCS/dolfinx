// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-10-21
// Last changed: 2006-12-12

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/File.h>
#include <dolfin/Form.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Form::Form(uint num_functions)
  : num_functions(num_functions), c(0), block(0)
{
  // Initialize list of functions
  if ( num_functions > 0 )
  {
    // Reserve list of functions
    functions.clear();
    functions.reserve(num_functions);
    for (uint i = 0; i < num_functions; i++)
      functions.push_back(0);

    // Reserve list of elements
    elements.clear();
    elements.reserve(num_functions);
    for (uint i = 0; i < num_functions; i++)
      elements.push_back(0);

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

  // Delete block
  if ( block )
    delete [] block;
}
//-----------------------------------------------------------------------------
void Form::initFunction(uint i, Function& f, FiniteElement* element)
{
  // Set finite element for function, but only for discrete functions
  if ( f.type() == Function::discrete )
    f.attach(*element);

  // Add function
  functions[i] = &f;

  // Add element (and delete old if any)
  if ( elements[i] )
    delete elements[i];
  elements[i] = element;

  // Initialize coefficients (and delete old if any)
  if ( c[i] )
    delete c[i];
  c[i] = new real[element->spacedim()];
  for (uint j = 0; j < element->spacedim(); j++)
    c[i][j] = 0.0;
}
//-----------------------------------------------------------------------------
void Form::update(Cell& cell, AffineMap& map)
{
  // Update coefficients
  updateCoefficients(cell, map);

  // Update local data structures
  updateLocalData();
}
//-----------------------------------------------------------------------------
void Form::update(Cell& cell, AffineMap& map, uint facet)
{
  // Update coefficients
  updateCoefficients(cell, map, facet);

  // Update local data structures
  updateLocalData();
}
//-----------------------------------------------------------------------------
void Form::update(Cell& cell0, Cell& cell1,
                  AffineMap& map0, AffineMap& map1,
                  uint facet0, uint facet1)
{
  cout << "Updating form in Form" << endl;

  // Update coefficients
  updateCoefficients(cell0, cell1, map0, map1, facet0, facet1);

  // Update local data structures
  updateLocalData();
}
//-----------------------------------------------------------------------------
void Form::updateCoefficients(Cell& cell, AffineMap& map)
{
  dolfin_assert(num_functions == functions.size());

  // Interpolate all functions to the current element
  for (uint i = 0; i < num_functions; i++)
  {
    dolfin_assert(functions[i]);
    functions[i]->interpolate(c[i], cell, map, *elements[i]);
  }
}
//-----------------------------------------------------------------------------
void Form::updateCoefficients(Cell& cell, AffineMap& map, uint facet)
{
  dolfin_assert(num_functions == functions.size());

  // Interpolate all functions to the current element
  for (uint i = 0; i < num_functions; i++)
  {
    dolfin_assert(functions[i]);
    functions[i]->interpolate(c[i], cell, map, *elements[i], facet);
  }
}
//-----------------------------------------------------------------------------
void Form::updateCoefficients(Cell& cell0, Cell& cell1,
                              AffineMap& map0, AffineMap& map1,
                              uint facet0, uint facet1)
{
  // FIXME: This is a temporary solution. We need to double the size
  // FIXME: of the coefficient arrays. When we move to UFC, the coefficients
  // FIXME: are given as an argument so this will be simpler to solve then.

  dolfin_assert(num_functions == functions.size());

  cout << "Interpolating function on interior facet" << endl;
  cout << "    cell0.index() = " << cell0.index() << "  facet0 = " << facet0 << endl;
  cout << "    cell1.index() = " << cell1.index() << "  facet1 = " << facet1 << endl;

  // Temporary coefficient arrays
  real** c0 = new real* [num_functions];
  real** c1 = new real* [num_functions];
  for (uint i = 0; i < num_functions; i++)
  {
    uint n = elements[i]->spacedim();
    c0[i] = new real[n];
    c1[i] = new real[n];
    for (uint j = 0; j < n; j++)
    {
      c0[i][j] = 0.0;
      c1[i][j] = 0.0;
    }
  }

  // Resize standard coefficient array
  if ( c )
  {
    for (uint i = 0; i < num_functions; i++)
      delete [] c[i];
    delete [] c;
  }
  c = new real* [num_functions];
  for (uint i = 0; i < num_functions; i++)
  {
    uint n = elements[i]->spacedim();
    c[i] = new real[2*n];
    for (uint j = 0; j < 2*n; j++)
      c[i][j] = 0.0;
  }
  
  cout << endl;

  // Interpolate on cell 0
  for (uint i = 0; i < num_functions; i++)
  {
    dolfin_assert(functions[i]);
    functions[i]->interpolate(c0[i], cell0, map0, *elements[i], facet0);

    for (uint j = 0; j < elements[0]->spacedim(); j++)
      dolfin_info("  c0[%d][%d] = %g", i, j, c0[i][j]);
  }

  cout << endl;

  // Interpolate on cell 1
  for (uint i = 0; i < num_functions; i++)
  {
    dolfin_assert(functions[i]);
    functions[i]->interpolate(c1[i], cell1, map1, *elements[i], facet1);

    for (uint j = 0; j < elements[0]->spacedim(); j++)
      dolfin_info("  c1[%d][%d] = %g", i, j, c1[i][j]);
  }

  cout << endl;

  // Copy values to large array
  for (uint i = 0; i < num_functions; i++)
  {
    uint n = elements[i]->spacedim();
    for (uint j = 0; j < n; j++)
    {
      c[i][j] = c0[i][j];
      c[i][j + n] = c1[i][j];
    }
  }

  for (uint i = 0; i < 2*elements[0]->spacedim(); i++)
  {
    dolfin_info("  c[0][%d] = %g", i, c[0][i]);
  }
  
  // Delete temporary storage
  for (uint i = 0; i < num_functions; i++)
  {
    delete [] c0[i];
    delete [] c1[i];
  }
  delete [] c0;
  delete [] c1;
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
