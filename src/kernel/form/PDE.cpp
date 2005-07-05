// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2005

#include <dolfin/dolfin_log.h>
#include <dolfin/PDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PDE::PDE() : bilinear(0), linear(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PDE::PDE(BilinearForm& a, LinearForm& L) : bilinear(&a), linear(&L)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PDE::~PDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BilinearForm& PDE::a()
{
  if ( !bilinear )
    dolfin_error("Bilinear form is not specified.");
  return *bilinear;
}
//-----------------------------------------------------------------------------
LinearForm& PDE::L()
{
  if ( !linear )
    dolfin_error("Linear form is not specified.");
  return *linear;
}
//-----------------------------------------------------------------------------
