// Copyright (C) 2005-2006 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-29
// Last changed: 2009-09-08

#include <cmath>
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
  timesteps = new real[ode.size()];
  residuals = new real[ode.size()];
  ktmp = new real[ode.size()];

  // Initialize local array for quadrature
  f = new real[method.qsize()];
  for (unsigned int i = 0; i < method.qsize(); i++)
    f[i] = 0.0;

  // Set initial time steps
  real k0 = static_cast<double>(ode.parameters["initial_time_step"]);
  if ( kfixed )
  {
    for (uint i = 0; i < ode.size(); i++)
    {
      timesteps[i] = ode.timestep(0.0, i, k0);
    }
  }
  else
  {
    real k = k0;
    if ( k > _kmax )
    {
      k = _kmax;
      warning("Initial time step larger than maximum time step, using k = %.3e.", to_double(k));
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
real MultiAdaptivity::timestep(uint i) const
{
  return timesteps[i];
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::residual(uint i) const
{
  return residuals[i];
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::update(MultiAdaptiveTimeSlab& ts, real t, bool first)
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
  compute_residuals(ts);

  // Accept if error is small enough
  if ( emax <= tol )
    _accept = true;

  // Update time steps for all components
  for (uint i = 0; i < ode.size(); i++)
  {
    // Previous time step
    const real k0 = timesteps[i];

    // Include dynamic safety factor
    real used_tol = safety*tol;

    // Compute new time step
    real k = method.timestep(residuals[i], used_tol, k0, _kmax);

    // Apply time step regulation
    k = Controller::update_harmonic(k, timesteps[i], _kmax);

    // Make sure to decrease the time step if not accepted
    if ( !_accept )
    {
      k = real_min(k, 0.9*k0);
    }

    // Save time step for component
    timesteps[i] = k;
  }

  // Propagate time steps according to dependencies
  propagate_dependencies();
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::compute_residuals(MultiAdaptiveTimeSlab& ts)
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
    e1 = ts.cover_slab(s, e0);

    // Get data for sub slab
    const real a = ts.sa[s];
    const real b = ts.sb[s];
    const real k = b - a;

    // Iterate over all elements in current sub slab
    for (uint e = e0; e < e1; e++)
    {
      // Get element data
      const uint i = ts.ei[e];

      // Get initial value for element
      const int ep = ts.ee[e];
      const real x0 = ( ep != -1 ? ts.jx[ep*method.nsize() + method.nsize() - 1] : ts.u0[i] );

      // Evaluate right-hand side at quadrature points of element
      if ( method.type() == Method::cG )
	ts.cg_feval(f, s, e, i, a, b, k);
      else
	ts.dg_feval(f, s, e, i, a, b, k);

      // Update maximum residual for component
      const real r = method.residual(x0, ts.jx + j, f[method.nsize()], k);
      residuals[i] = real_max(residuals[i], real_abs(r));

      // Update maximum residual and error
      rmax = real_max(rmax, r);
      emax = real_max(emax, method.error(k, r));

      // Update dof
      j += method.nsize();
    }

    // Step to next sub slab
    e0 = e1;
  }
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::propagate_dependencies()
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
    const real k = ktmp[i];

    // Propagate time step to dependencies
    const std::vector<uint>& deps = ode.dependencies[i];
    for (uint pos = 0; pos < deps.size(); pos++)
      timesteps[deps[pos]] = real_min(timesteps[deps[pos]], k);
  }
}
//-----------------------------------------------------------------------------
