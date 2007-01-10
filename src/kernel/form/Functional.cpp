// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-09-18
// Last changed: 2006-12-12

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
void Functional::updateLocalData()
{
  // Initialize block
  if ( !block )
    block = new real[1];
  block[0] = 0.0;
}
//-----------------------------------------------------------------------------
