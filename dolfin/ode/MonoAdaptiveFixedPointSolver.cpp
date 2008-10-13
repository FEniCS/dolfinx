// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2008-10-07

#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/parameters.h>
#include "Alloc.h"
#include "Method.h"
#include "MonoAdaptiveTimeSlab.h"
#include "MonoAdaptiveFixedPointSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveFixedPointSolver::MonoAdaptiveFixedPointSolver
(MonoAdaptiveTimeSlab& timeslab)
  : TimeSlabSolver(timeslab), ts(timeslab), xold(0),
    stabilize(ode.get("ODE fixed-point stabilize")), mi(0), li(0), ramp(1.0)
{
  double tmp = ode.get("ODE fixed-point stabilization ramp");
  rampfactor = tmp;

  tmp = ode.get("ODE fixed-point damping");
  alpha = tmp;

  // Initialize old values at right end-point
  xold = new real[ts.N];
  for (uint i = 0; i < ts.N; i++)
    xold[i] = 0.0;
}
//-----------------------------------------------------------------------------
MonoAdaptiveFixedPointSolver::~MonoAdaptiveFixedPointSolver()
{
  delete [] xold;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveFixedPointSolver::iteration(real tol, uint iter,
                                               real d0, real d1)
{
  //   real K = ts.endtime() - ts.starttime();
  real alpha_orig = alpha;
  if(stabilize)
  {
    if (iter == 0)
    {
      ramp = 1.0;
      mi = 0;
      li = 0;
    }
    
    if (iter == 0 || (d1 > d0 && li == 0))
    {
      ramp = 1.0;
      mi = ode.get("ODE fixed-point stabilization m");
      //mi = (int)ceil(log10(K * 1.0e4));
    }  
    
    if (mi == 0 && li == 0)
    {
      // Choose number of ramping iterations
      li = ode.get("ODE fixed-point stabilization l");
    }
    
    if (mi == 0)
    {
      // Ramping
      ramp = ramp * rampfactor;
    }

    alpha *= ramp;
  }

  // Compute size of time step
  const real k = ts.length();

  // Save old values
  const uint xoffset = (method.nsize() - 1) * ts.N;
  ts.copy(ts.x, xoffset, xold, 0, ts.N);

  // Save norm of old solution
  xnorm = 0.0;
  for (uint j = 0; j < ts.nj; j++)
    xnorm = max(xnorm, abs(ts.x[j]));

  // Evaluate right-hand side at all quadrature points
  for (uint m = 0; m < method.qsize(); m++)
    ts.feval(m);

  // Update the values at each stage
  for (uint n = 0; n < method.nsize(); n++)
  {
    const uint noffset = n * ts.N;

    // Reset values to initial data
    for (uint i = 0; i < ts.N; i++)
      ts.x[noffset + i] += alpha*(ts.u0[i] - ts.x[noffset+i]);

    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
	ts.x[noffset + i] += alpha*tmp*ts.fq[moffset + i];
    }
  }
  
  // Compute size of increment
  real max_increment = 0.0;
  for (uint i = 0; i < ts.N; i++)
  {
    const real increment = abs(ts.x[xoffset + i] - xold[i]);
    if ( increment > max_increment )
      max_increment = increment;
  }

  if (stabilize)
  {
    alpha = alpha_orig;

    if (mi > 0)
      mi -= 1;
    if (li > 0)
      li -= 1;
  }

  return max_increment;
}
//-----------------------------------------------------------------------------
dolfin::uint MonoAdaptiveFixedPointSolver::size() const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
