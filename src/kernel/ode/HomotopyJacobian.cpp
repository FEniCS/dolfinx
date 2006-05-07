// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_log.h>
#include <dolfin/ComplexODE.h>
#include <dolfin/Vector.h>
#include <dolfin/HomotopyJacobian.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
HomotopyJacobian::HomotopyJacobian(ComplexODE& ode, Vector& u) 
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
void HomotopyJacobian::mult(const Vector& x, Vector& y) const
{
  // Get arrays (assumes uniprocessor case)
  const real* uu = u.array();
  const real* xx = x.array();
  real* yy = y.array();

  // Compute y = A*x
  ode.J(xx, yy, uu, 0.0);

  // Restore arrays
  u.restore(uu);
  x.restore(xx);
  y.restore(yy);
}
//-----------------------------------------------------------------------------

#endif
