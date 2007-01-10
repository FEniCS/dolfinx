// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-28
// Last changed: 2006-08-21

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_parameter.h>
#include <dolfin/Alloc.h>
#include <dolfin/Method.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>
#include <dolfin/MonoAdaptiveFixedPointSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveFixedPointSolver::MonoAdaptiveFixedPointSolver
(MonoAdaptiveTimeSlab& timeslab)
  : TimeSlabSolver(timeslab), ts(timeslab), xold(0),
    alpha(get("ODE fixed-point damping")),
    stabilize(get("ODE fixed-point stabilize")), mi(0), li(0), ramp(1.0),
    rampfactor(get("ODE fixed-point stabilization ramp"))
{
  // Initialize old values at right end-point
  xold = new real[ts.N];
  for (uint i = 0; i < ts.N; i++)
    xold[i] = 0.0;
}
//-----------------------------------------------------------------------------
MonoAdaptiveFixedPointSolver::~MonoAdaptiveFixedPointSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real MonoAdaptiveFixedPointSolver::iteration(real tol, uint iter,
					     real d0, real d1)
{
//   real K = ts.endtime() - ts.starttime();

  real alpha_orig = alpha;

  if(stabilize)
  {

    if(iter == 0)
    {
      ramp = 1.0;
      mi = 0;
      li = 0;
    }
    
    if(iter == 0 || (d1 > d0 && li == 0))
    {
      // stabilize
      
      ramp = 1.0;
      mi = get("ODE fixed-point stabilization m");
      //mi = (int)ceil(log10(K * 1.0e4));

      //cout << "stabilize: " << mi << endl;
    }  
    
    if(mi == 0 && li == 0)
    {
      // Choose number of ramping iterations
      li = get("ODE fixed-point stabilization l");

      //cout << "ramp: " << li << endl;
    }
    
    if(mi == 0)
    {
      // ramping
      
      ramp = ramp * rampfactor;
    }
    

    alpha *= ramp;
  }

  // Compute size of time step
  const real k = ts.length();

  // Save old values
  const uint xoffset = (method.nsize() - 1) * ts.N;
  ts.copy(ts.x, xoffset, xold, 0, ts.N);

  // Evaluate right-hand side at all quadrature points
  for (uint m = 0; m < method.qsize(); m++)
    ts.feval(m);

  // Update the values at each stage
  for (uint n = 0; n < method.nsize(); n++)
  {
    const uint noffset = n * ts.N;

    // Reset values to initial data
    for (uint i = 0; i < ts.N; i++)
      ts.x(noffset + i) += alpha*(ts.u0(i) - ts.x(noffset+i));

    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
	ts.x(noffset + i) += alpha*tmp*ts.fq[moffset + i];
    }
  }
  
  // Compute size of increment
  real max_increment = 0.0;
  for (uint i = 0; i < ts.N; i++)
  {
    const real increment = fabs(ts.x(xoffset + i) - xold[i]);
    if ( increment > max_increment )
      max_increment = increment;
  }

  if(stabilize)
  {
    alpha = alpha_orig;

    if(mi > 0)
      mi -= 1;
    if(li > 0)
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
