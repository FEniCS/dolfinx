// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/LinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearForm::LinearForm(const NewFiniteElement& element) : Form(element)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LinearForm::~LinearForm()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool LinearForm::interior(real* b) const
{
  // The default version returns false, which means that the form does
  // not contain any integrals over the interior of the domain.
  return false;
}
//-----------------------------------------------------------------------------
bool LinearForm::boundary(real* b) const
{
  // The default version returns false, which means that the form does
  // not contain any integrals over the boundary of the domain.
  return false;
}
//-----------------------------------------------------------------------------
