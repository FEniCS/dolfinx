// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewFiniteElement.h>
#include <dolfin/BilinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BilinearForm::BilinearForm() : Form()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BilinearForm::~BilinearForm()
{
  // Do nothing
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
