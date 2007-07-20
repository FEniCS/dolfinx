// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-27
// Last changed: 2006-08-21

#include <dolfin/dolfin_math.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveNewtonSolver.h>
#include <dolfin/UpdatedMultiAdaptiveJacobian.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UpdatedMultiAdaptiveJacobian::UpdatedMultiAdaptiveJacobian
(MultiAdaptiveNewtonSolver& newton, MultiAdaptiveTimeSlab& timeslab)
  : TimeSlabJacobian(timeslab), newton(newton), ts(timeslab),  h(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UpdatedMultiAdaptiveJacobian::~UpdatedMultiAdaptiveJacobian()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint UpdatedMultiAdaptiveJacobian::size(uint dim) const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
void UpdatedMultiAdaptiveJacobian::mult(const uBlasVector& x,
					uBlasVector& y) const
{
  // Compute product by the approximation y = J(u) x = (F(u + hx) - F(u)) / h.
  // Since Feval() compute -F rather than F, we compute according to
  //
  //     y = J(u) x = (-F(u - hx) - (-F(u))) / h

  // Update values, u <-- u - hx
  for (unsigned int j = 0; j < ts.nj; j++)
    ts.jx[j] -= h*x(j);

  // Compute -F(u - hx)
  newton.Feval(y);

  // Restore values, u <-- u + hx
  for (unsigned int j = 0; j < ts.nj; j++)
    ts.jx[j] += h*x(j);

  // Compute difference, using already computed -F(u)
  y -= newton. b;
  y /= h;
}
//-----------------------------------------------------------------------------
void UpdatedMultiAdaptiveJacobian::init()
{
  // Compute size of increment
  real umax = 0.0;
  for (unsigned int i = 0; i < ts.N; i++)
    umax = std::max(umax, std::abs(ts.u0(i)));
  h = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);
}
//-----------------------------------------------------------------------------
