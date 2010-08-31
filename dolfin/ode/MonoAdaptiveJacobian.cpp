// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2009-03-07

#include <dolfin/common/real.h>
#include <dolfin/common/timing.h>
#include <dolfin/math/dolfin_math.h>
#include "ODE.h"
#include "Method.h"
#include "MonoAdaptiveTimeSlab.h"
#include "MonoAdaptiveJacobian.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveJacobian::MonoAdaptiveJacobian(MonoAdaptiveTimeSlab& timeslab,
					   bool implicit, bool piecewise)
  : TimeSlabJacobian(timeslab), ts(timeslab),
    implicit(implicit), piecewise(piecewise), xx(ts.N), yy(ts.N)
{
  //Do nothing
  //xx = new real[ts.N];
  //yy = new real[ts.N];

  //real_zero(ts.N, xx);
  //real_zero(ts.N, yy);
  xx.zero();
  yy.zero();
}
//-----------------------------------------------------------------------------
MonoAdaptiveJacobian::~MonoAdaptiveJacobian()
{
  //delete [] xx;
  //delete [] yy;
}
//-----------------------------------------------------------------------------
dolfin::uint MonoAdaptiveJacobian::size(uint dim) const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveJacobian::mult(const uBLASVector& x, uBLASVector& y) const
{
  // Start with y = x, accounting for the derivative dF_j/dx_j = 1
  if (!implicit)
    y = x;

  // Compute size of time step
  const real a = ts.starttime();
  const real k = ts.length();

  // Compute product y = Mx for each stage for implicit system
  if (implicit)
  {
    // Iterate over stages
    for (uint n = 0; n < method.nsize(); n++)
    {
      const uint noffset = n * ts.N;

      // Copy values to xx
      real_copy(x, noffset, xx);
      //ts.copy(x, noffset, xx);

      // Do multiplication
      if (piecewise)
      {
        ode.M(xx, yy, ts.u0, a);
      }
      else
      {
        const real t = a + method.npoint(n) * k;
        //ts.copy(ts.x, noffset, ts.u);
        real_copy(ts.x, noffset, ts.u);
        ode.M(xx, yy, ts.u, t);
      }

      // Copy values from yy
      real_copy(yy, y, noffset);
    }
  }

  // Iterate over the stages
  for (uint n = 0; n < method.nsize(); n++)
  {
    const real t = a + method.npoint(n) * k;
    const uint noffset = n * ts.N;

    /*
    // Compute yy = df/du * x for current stage
    for (uint i = 0; i < ts.N; i++)
    {
      double sum = 0.0;
      const std::vector<uint>& deps = ode.dependencies[i];
      const uint Joffset = Jindices[i];
      for (uint pos = 0; pos < deps.size(); pos++)
      {
        const uint j = deps[pos];
        sum += Jvalues[Joffset + pos] * xxx[noffset + j];
      }
      yy[i] = sum;
    }
    */

    // Copy values to xx and u
    real_copy(x, noffset, xx);
    //ts.copy(x, noffset, xx);
    real_copy(ts.x, noffset, ts.u);
    //ts.copy(ts.x, noffset, ts.u);

    // Compute z = df/du * x for current stage
    ode.J(xx, yy, ts.u, t);

    // Add z with correct weights to y
    for (uint m = 0; m < method.nsize(); m++)
    {
      const uint moffset = m * ts.N;

      // Get correct weight
      real w = 0.0;
      if (method.type() == Method::cG)
        w = - k * method.nweight(m, n + 1);
      else
        w = - k * method.nweight(m, n);

      // Add w*yy to y
      for (uint i = 0; i < ts.N; i++)
	// Note: Precision lost if working with GMP
        y[moffset + i] += to_double((w * yy[i]));
    }
  }
}
//-----------------------------------------------------------------------------
