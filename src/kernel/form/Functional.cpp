// Copyright (C) 2005 Johan Hoffman. 
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-02

#include <dolfin/Functional.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Functional::Functional(uint num_functions) : Form(num_functions), _test(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Functional::~Functional()
{
  if ( _test ) delete _test;
}
//-----------------------------------------------------------------------------
void Functional::eval(real block[], const AffineMap& map) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Functional::eval(real block[], const AffineMap& map, uint segment) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const FiniteElement& Functional::test() const
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
