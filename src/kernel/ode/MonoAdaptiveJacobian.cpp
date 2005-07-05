// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-28
// Last changed: 2005

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
    implicit(implicit), piecewise(piecewise)
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
  const real* xx = x.array();
  real* yy = y.array();
  const real* uu = ts.x.array();

  // Temporary data array used to store multiplications
  real* z = ts.tmp();

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

      // Do multiplication
      if ( piecewise )
      {
	ode.M(xx + noffset, z, ts.u0, a);
      }
      else
      {
	const real t = a + method.npoint(n) * k;
	ode.M(xx + noffset, z, uu + noffset, t);
      }
      
      // Copy values
      for (uint i = 0; i < ts.N; i++)
	yy[noffset + i] = z[i];
    }
  }

  // Iterate over the stages
  for (uint n = 0; n < method.nsize(); n++)
  {
    const real t = a + method.npoint(n) * k;
    const uint noffset = n * ts.N;

    /*
    // Compute z = df/du * x for current stage
    for (uint i = 0; i < ts.N; i++)
    {
      real sum = 0.0;
      const Array<uint>& deps = ode.dependencies[i];
      const uint Joffset = Jindices[i];
      for (uint pos = 0; pos < deps.size(); pos++)
      {
	const uint j = deps[pos];
	sum += Jvalues[Joffset + pos] * xx[noffset + j];
      }
      z[i] = sum;
    }
    */

    // Compute z = df/du * x for current stage
    ode.J(xx + noffset, z, uu + noffset, t);
    
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

      // Add w * z to y
      for (uint i = 0; i < ts.N; i++)
	yy[moffset + i] += w * z[i];
    }
  }

  // Restore data arrays
  x.restore(xx);
  y.restore(yy);
  ts.x.restore(uu);
}
//-----------------------------------------------------------------------------
