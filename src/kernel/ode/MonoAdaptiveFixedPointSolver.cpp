// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Alloc.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>
#include <dolfin/MonoAdaptiveFixedPointSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveFixedPointSolver::MonoAdaptiveFixedPointSolver
(MonoAdaptiveTimeSlab& timeslab) : TimeSlabSolver(timeslab), ts(timeslab)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MonoAdaptiveFixedPointSolver::~MonoAdaptiveFixedPointSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real MonoAdaptiveFixedPointSolver::iteration()
{
  // Compute size of time step
  const real k = ts.length();

  // Get array of values
  real* xx = ts.x.array();

  // Evaluate right-hand side at all quadrature points
  for (uint m = 0; m < method.qsize(); m++)
    ts.feval(m);

  // Update the values at each stage
  for (uint n = 0; n < method.nsize(); n++)
  {
    const uint noffset = n * ts.N;

    // Reset values to initial data
    for (uint i = 0; i < ts.N; i++)
      xx[noffset + i] = ts.u0[i];
    
    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
	xx[noffset + i] += tmp * ts.f[moffset + i];
    }
  }

  // Restore array
  ts.x.restore(xx);

  return 0.0;
}
//-----------------------------------------------------------------------------
