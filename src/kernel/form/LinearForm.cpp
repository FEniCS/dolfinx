// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/LinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearForm::LinearForm(uint nfunctions) : Form(nfunctions), _test(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LinearForm::~LinearForm()
{
  if ( _test ) delete _test;
}
//-----------------------------------------------------------------------------
bool LinearForm::interior(real* block) const
{
  // The default version returns false, which means that the form does
  // not contain any integrals over the interior of the domain.
  return false;
}
//-----------------------------------------------------------------------------
bool LinearForm::boundary(real* block) const
{
  // The default version returns false, which means that the form does
  // not contain any integrals over the boundary of the domain.
  return false;
}
//-----------------------------------------------------------------------------
const NewFiniteElement& LinearForm::test() const
{
  dolfin_assert(_test); // Should be created by child class
  return *_test;
}
//-----------------------------------------------------------------------------
