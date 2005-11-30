// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-05-28
// Last changed: 2005-11-29

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
void BilinearForm::eval(real block[], const AffineMap& map, uint segment) const
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FiniteElement& BilinearForm::test()
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
FiniteElement& BilinearForm::trial()
{
  dolfin_assert(_trial); // Should be created by child class
  return *_trial;
}
//-----------------------------------------------------------------------------
