// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin/FiniteElement.h>
#include <dolfin/BilinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BilinearForm::BilinearForm(uint nfunctions)
  : Form(nfunctions), _test(0), _trial(0)
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
bool BilinearForm::interior(real* block) const
{
  // The default version returns false, which means that the form does
  // not contain any integrals over the interior of the domain.
  return false;
}
//-----------------------------------------------------------------------------
bool BilinearForm::boundary(real* block) const
{
  // The default version returns true, which means that the form does
  // not contain any integrals over the boundary of the domain.
  return false;
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
