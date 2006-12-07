// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-09-18
// Last changed: 2006-12-07

#include <dolfin/Functional.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Functional::Functional(uint num_functions) : Form(num_functions)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Functional::~Functional()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Functional::update(AffineMap& map)
{
  // Update coefficients
  updateCoefficients(map);

  // Update local data structures
  updateLocalData();
}
//-----------------------------------------------------------------------------
void Functional::updateLocalData()
{
  // Initialize block
  if ( !block )
    block = new real[1];
  block[0] = 0.0;
}
//-----------------------------------------------------------------------------
