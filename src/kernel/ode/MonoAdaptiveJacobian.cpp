// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-28
// Last changed: 2006-07-05

#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_math.h>
#include <dolfin/ODE.h>
#include <dolfin/Vector.h>
#include <dolfin/Method.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>
#include <dolfin/MonoAdaptiveJacobian.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveJacobian::MonoAdaptiveJacobian(MonoAdaptiveTimeSlab& timeslab,
					   bool implicit, bool piecewise)
  : TimeSlabJacobian(timeslab), ts(timeslab),
    implicit(implicit), piecewise(piecewise), xx(timeslab.N), yy(timeslab.N)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MonoAdaptiveJacobian::~MonoAdaptiveJacobian()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MonoAdaptiveJacobian::mult(const Vector& x, Vector& y) const
{
  // Start with y = x, accounting for the derivative dF_j/dx_j = 1
  if ( !implicit )
    y = x;

  // Get data arrays (assumes uniprocessor case)
  const real* xxx = x.array();
  real* yyy = y.array();
  const real* uu = ts.x.array();

  // Compute size of time step
  const real a = ts.starttime();
  const real k = ts.length();

  // Compute product y = Mx for each stage for implicit system
  if ( implicit )
  {
    // Iterate over stages
    for (uint n = 0; n < method.nsize(); n++)
    {
      const uint noffset = n * ts.N;

      // Copy values to xx
      ts.copy(xxx, noffset, xx, 0, ts.N);

      // Do multiplication
      if ( piecewise )
      {
	ts.copy(ts.u0, 0, ts.u, 0, ts.N);
	ode.M(xx, yy, ts.u, a);
      }
      else
      {
	const real t = a + method.npoint(n) * k;
	ts.copy(uu, noffset, ts.u, 0, ts.N);
	ode.M(xx, yy, ts.u, t);
      }
      
      // Copy values from yy
      ts.copy(yy, 0, yyy, noffset, ts.N);
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
      real sum = 0.0;
      const Array<uint>& deps = ode.dependencies[i];
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
    ts.copy(xxx, noffset, xx, 0, ts.N);
    ts.copy(uu, noffset, ts.u, 0, ts.N);

    // Compute z = df/du * x for current stage
    ode.J(xx, yy, ts.u, t);

    // Add z with correct weights to y
    for (uint m = 0; m < method.nsize(); m++)
    {
      const uint moffset = m * ts.N;

      // Get correct weight
      real w = 0.0;
      if ( method.type() == Method::cG )
	w = - k * method.nweight(m, n + 1);
      else
	w = - k * method.nweight(m, n);

      // Add w*yy to y
      for (uint i = 0; i < ts.N; i++)
	yyy[moffset + i] += w * yy[i];
    }
  }

  // Restore data arrays
  x.restore(xxx);
  y.restore(yyy);
  ts.x.restore(uu);
}
//-----------------------------------------------------------------------------

#endif
