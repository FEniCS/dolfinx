// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Dual.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Dual::Dual(ODE& ode, Function& u) : ODE(ode.size()), ode(ode), u(u)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Dual::~Dual()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real Dual::u0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Dual::f(const Vector& u, real t, unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
