// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewPDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewPDE::NewPDE()
{
  det = 0.0;

  g00 = 0.0; g01 = 0.0; g02 = 0.0;
  g10 = 0.0; g11 = 0.0; g12 = 0.0;
  g20 = 0.0; g21 = 0.0; g22 = 0.0;
}
//-----------------------------------------------------------------------------
NewPDE::~NewPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewPDE::lhs(NewArray< NewArray<real> >& A)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewPDE::rhs(NewArray<real>& b)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
