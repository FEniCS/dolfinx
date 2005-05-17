// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

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
void LinearForm::eval(real block[], const AffineMap& map, uint boundary) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const FiniteElement& LinearForm::test() const
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
