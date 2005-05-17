// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin/FiniteElement.h>
#include <dolfin/BilinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BilinearForm::BilinearForm(uint num_functions)
  : Form(num_functions), _test(0), _trial(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BilinearForm::~BilinearForm()
{
  if ( _test ) delete _test;
  if ( _trial ) delete _trial;
}
//-----------------------------------------------------------------------------
void BilinearForm::eval(real block[], const AffineMap& map) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void BilinearForm::eval(real block[], const AffineMap& map, uint boundary) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const FiniteElement& BilinearForm::test() const
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
const FiniteElement& BilinearForm::trial() const
{
  dolfin_assert(_trial); // Should be created by child class
  return *_trial;
}
//-----------------------------------------------------------------------------
