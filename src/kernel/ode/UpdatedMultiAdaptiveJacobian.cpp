// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_math.h>
#include <dolfin/ODE.h>
#include <dolfin/Vector.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveNewtonSolver.h>
#include <dolfin/UpdatedMultiAdaptiveJacobian.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UpdatedMultiAdaptiveJacobian::UpdatedMultiAdaptiveJacobian
(MultiAdaptiveNewtonSolver& newton, MultiAdaptiveTimeSlab& timeslab)
  : TimeSlabJacobian(timeslab), newton(newton), ts(timeslab), tmp(0), nj(0), h(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UpdatedMultiAdaptiveJacobian::~UpdatedMultiAdaptiveJacobian()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void UpdatedMultiAdaptiveJacobian::update()
{
  // Check if we need to reallocate the temporary array
  if ( ts.nj > nj )
  {
    if ( tmp )
      delete [] tmp;
    tmp = new real[ts.nj];
  }
  
  // Save size of system
  nj = ts.nj;

  // Compute size of increment
  real umax = 0.0;
  for (unsigned int i = 0; i < ts.N; i++)
    umax = std::max(umax, std::abs(ts.u0[i]));
  h = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);
}
//-----------------------------------------------------------------------------
void UpdatedMultiAdaptiveJacobian::mult(const Vector& x, Vector& y) const
{
  // Compute product by the approximation y = J(u) x = (F(u + hx) - F(u)) / h.
  // Since Feval() compute -F rather than F, we compute according to
  //
  //     y = J(u) x = (-F(u - hx) - (-F(u))) / h

  // Get data arrays (assumes uniprocessor case)
  const real* xx = x.array();
  real* yy = y.array();
  
  // Update values, u <-- u - hx
  for (unsigned int j = 0; j < nj; j++)
    ts.jx[j] -= h*xx[j];

  // Compute -F(u - hx)
  newton.Feval(tmp);

  // Restore values, u <-- u + hx
  for (unsigned int j = 0; j < nj; j++)
    ts.jx[j] += h*xx[j];

  // Compute difference, using already computed -F(u)
  real* bb = newton.b.array(); // Assumes uniprocessor case
  for (unsigned int j = 0; j < nj; j++)
    yy[j] = (tmp[j] - bb[j]) / h;
  newton.b.restore(bb);

  // Restore data arrays
  x.restore(xx);
  y.restore(yy);
}
//-----------------------------------------------------------------------------

#endif
