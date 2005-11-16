// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005-11-04

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptivity::MultiAdaptivity(const ODE& ode, const Method& method)
  : Adaptivity(ode, method)
{
  // Initialize time steps
  real k0 = dolfin_get("initial time step");
  timesteps = new real[ode.size()];
  if ( kfixed )
  {
    for (uint i = 0; i < ode.size(); i++)
      timesteps[i] = ode.timestep(0.0, i, k0);
  }
  else
  {
    real k = k0;
    if ( k > _kmax )
    {
      k = _kmax;
      dolfin_warning1("Initial time step larger than maximum time step, using k = %.3e.", k);
    }
    for (uint i = 0; i < ode.size(); i++)
      timesteps[i] = k;
  }
}
//-----------------------------------------------------------------------------
MultiAdaptivity::~MultiAdaptivity()
{
  if ( timesteps ) delete [] timesteps;
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::timestep(uint i) const
{
  return timesteps[i];
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::updateStart()
{
  // Will remain true if solution can be accepted for all components
  _accept = true;
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::updateComponent(uint i, real k0, real r,
				      const Method& method, real t)
{
  // Check if time step is fixed
  if ( kfixed )
  {
    timesteps[i] = ode.timestep(t, i, k0);
    _accept = true;
    return;
  }

  // Compute local error estimate
  const real error = method.error(k0, r);
  
  // Compute new time step
  real k = method.timestep(r, safety*tol, k0, _kmax);
  k = Controller::updateHarmonic(k, timesteps[i], _kmax);
  
  // Check if time step can be accepted
  if ( error > tol )
  {
    k = std::min(k, 0.5*k0);
    _accept = false;
  }

  // Save time step for component
  timesteps[i] = k;
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::updateEnd(bool first)
{
  // Reduce overall size of time slab if this is the first time slab
  if ( first )
  {
    real K = 0.0;
    for (uint i = 0; i < ode.size(); i++)
      K = std::max(K, timesteps[i]);
 
    for (uint i = 0; i < ode.size(); i++)
      timesteps[i] = std::min(K, timesteps[i]);
  }
}
//-----------------------------------------------------------------------------
