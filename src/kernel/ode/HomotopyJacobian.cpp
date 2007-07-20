// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005
// Last changed: 2006-07-07

#include <dolfin/dolfin_log.h>
#include <dolfin/ComplexODE.h>
#include <dolfin/HomotopyJacobian.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
HomotopyJacobian::HomotopyJacobian(ComplexODE& ode, uBlasVector& u) 
  : ode(ode), u(u)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
HomotopyJacobian::~HomotopyJacobian()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint HomotopyJacobian::size(uint dim) const
{
  return u.size();
}
//-----------------------------------------------------------------------------
void HomotopyJacobian::mult(const uBlasVector& x, uBlasVector& y) const
{
  // Compute y = A*x
  ode.J(x, y, u, 0.0);
}
//-----------------------------------------------------------------------------

