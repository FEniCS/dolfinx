// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewFiniteElement.h>
#include <dolfin/BilinearForm.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BilinearForm::BilinearForm(const NewFiniteElement& element) : Form(element)
{
  // Set default (full) nonzero pattern
  for (unsigned int i = 0; i < element.spacedim(); i++)
    for (unsigned int j = 0; j < element.spacedim(); j++)
      nonzero.push_back(IndexPair(i, j));
}
//-----------------------------------------------------------------------------
BilinearForm::~BilinearForm()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool BilinearForm::interior(real** A) const
{
  // The default version returns false, which means that the form does
  // not contain any integrals over the interior of the domain.
  return false;
}
//-----------------------------------------------------------------------------
bool BilinearForm::boundary(real** A) const
{
  // The default version returns true, which means that the form does
  // not contain any integrals over the boundary of the domain.
  return false;
}
//-----------------------------------------------------------------------------
