// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/NewPDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewPDE::NewPDE() : bilinear(0), linear(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewPDE::NewPDE(BilinearForm& a, LinearForm& L) : bilinear(&a), linear(&L)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewPDE::~NewPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BilinearForm& NewPDE::a()
{
  if ( !bilinear )
    dolfin_error("Bilinear form is not specified.");
  return *bilinear;
}
//-----------------------------------------------------------------------------
LinearForm& NewPDE::L()
{
  if ( !linear )
    dolfin_error("Linear form is not specified.");
  return *linear;
}
//-----------------------------------------------------------------------------
