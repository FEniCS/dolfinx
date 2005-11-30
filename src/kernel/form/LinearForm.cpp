// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-05-28
// Last changed: 2005-11-29

#include <dolfin/LinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearForm::LinearForm(uint num_functions) : Form(num_functions), _test(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LinearForm::~LinearForm()
{
  if ( _test ) delete _test;
}
//-----------------------------------------------------------------------------
void LinearForm::eval(real block[], const AffineMap& map) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LinearForm::eval(real block[], const AffineMap& map, uint segment) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FiniteElement& LinearForm::test()
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
