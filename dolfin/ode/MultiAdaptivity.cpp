// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-29
// Last changed: 2006-04-20

#include <cmath>
#include <dolfin/parameter/parameters.h>
#include <dolfin/common/Array.h>
#include "ODE.h"
#include "Method.h"
#include "MultiAdaptiveTimeSlab.h"
#include "MultiAdaptivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptivity::MultiAdaptivity(const ODE& ode, const Method& method)
  : Adaptivity(ode, method), 
    timesteps(0), residuals(0), ktmp(0), f(0), rmax(0), emax(0)
{
  // Initialize time steps and residuals
  timesteps = new double[ode.size()];
  residuals = new double[ode.size()];
  ktmp = new double[ode.size()];

  // Initialize local array for quadrature
  f = new double[method.qsize()];
  for (unsigned int i = 0; i < method.qsize(); i++)
    f[i] = 0.0;

  // Set initial time steps
  double k0 = ode.get("ODE initial time step");
  if ( kfixed )
  {
    for (uint i = 0; i < ode.size(); i++)
    {
      timesteps[i] = ode.timestep(0.0, i, k0);
    }
  }
  else
  {
    double k = k0;
    if ( k > _kmax )
    {
      k = _kmax;
      warning("Initial time step larger than maximum time step, using k = %.3e.", k);
    }
    for (uint i = 0; i < ode.size(); i++)
      timesteps[i] = k;
  }

  // Initialize arrays
  for (uint i = 0; i < ode.size(); i++)
  {
    residuals[i] = 0.0;
    ktmp[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
MultiAdaptivity::~MultiAdaptivity()
{
  if ( timesteps ) delete [] timesteps;
  if ( residuals ) delete [] residuals;
  if ( ktmp ) delete [] ktmp;
}
//-----------------------------------------------------------------------------
double MultiAdaptivity::timestep(uint i) const
{
  return timesteps[i];
}
//-----------------------------------------------------------------------------
double MultiAdaptivity::residual(uint i) const
{
  return residuals[i];
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::update(MultiAdaptiveTimeSlab& ts, double t, bool first)
{
  _accept = false;

  // Check if time step is fixed
  if ( kfixed )
  {
    for (uint i = 0; i < ts.N; i++)
      timesteps[i] = ode.timestep(t, i, timesteps[i]);
   
    _accept = true;
    return;
  }

  // Compute maximum residuals for all components in time slab
  computeResiduals(ts);
  
  // Accept if error is small enough
  if ( emax <= tol )
    _accept = true;

  // Update time steps for all components
  for (uint i = 0; i < ode.size(); i++)
  {
    // Previous time step
    const double k0 = timesteps[i];
    
    // Include dynamic safety factor
    double used_tol = safety*tol;
    
    // Compute new time step
    double k = method.timestep(residuals[i], used_tol, k0, _kmax);

    // Apply time step regulation
    k = Controller::updateHarmonic(k, timesteps[i], _kmax);
    
    // Make sure to decrease the time step if not accepted
    if ( !_accept )
    {
      k = std::min(k, 0.9*k0);
    }
    
    // Save time step for component
    timesteps[i] = k;
  }

  // Propagate time steps according to dependencies
  propagateDependencies();
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::computeResiduals(MultiAdaptiveTimeSlab& ts)
{
  // Reset dof
  uint j = 0;

  // Reset elast
  for (uint i = 0; i < ts.N; i++)
    ts.elast[i] = -1;

  // Reset residuals
  for (uint i = 0; i < ts.N; i++)
    residuals[i] = 0.0;
  
  // Reset maximum local residual and error
  rmax = 0.0;
  emax = 0.0;

  // Iterate over all sub slabs
  uint e0 = 0;
  uint e1 = 0;
  for (uint s = 0; s < ts.ns; s++)
  {
    // Cover all elements in current sub slab
    e1 = ts.coverSlab(s, e0);
    
    // Get data for sub slab
    const double a = ts.sa[s];
    const double b = ts.sb[s];
    const double k = b - a;

    // Iterate over all elements in current sub slab
    for (uint e = e0; e < e1; e++)
    {
      // Get element data
      const uint i = ts.ei[e];

      // Get initial value for element
      const int ep = ts.ee[e];
      const double x0 = ( ep != -1 ? ts.jx[ep*method.nsize() + method.nsize() - 1] : ts.u0[i] );
      
      // Evaluate right-hand side at quadrature points of element
      if ( method.type() == Method::cG )
	ts.cGfeval(f, s, e, i, a, b, k);
      else
	ts.dGfeval(f, s, e, i, a, b, k);

      // Update maximum residual for component
      const double r = method.residual(x0, ts.jx + j, f[method.nsize()], k);
      residuals[i] = std::max(residuals[i], fabs(r));

      // Update maximum residual and error
      rmax = std::max(rmax, r);
      emax = std::max(emax, method.error(k, r));

      // Update dof
      j += method.nsize();
    }

    // Step to next sub slab
    e0 = e1;
  }
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::propagateDependencies()
{
  // This is a poor man's dual weighting function. For each component,
  // we look at all other components that the component in question
  // depends on and tell these other components to reduce their time
  // steps to the same level. A similar effect would be accomplished
  // by weighting with the proper weight from the dual solution, but
  // until we (re-)implement the solution of the dual, this seems to
  // be a good solution.

  // Don't propagate dependencies if not sparse
  if ( !ode.dependencies.sparse() )
    return;

  // Copy time steps
  for (uint i = 0; i < ode.size(); i++)
    ktmp[i] = timesteps[i];

  // Iterate over components
  for (uint i = 0; i < ode.size(); i++)
  {
    // Get time step for current component
    const double k = ktmp[i];

    // Propagate time step to dependencies
    const Array<uint>& deps = ode.dependencies[i];
    for (uint pos = 0; pos < deps.size(); pos++)
      timesteps[deps[pos]] = std::min(timesteps[deps[pos]], k);
  }
}
//-----------------------------------------------------------------------------
