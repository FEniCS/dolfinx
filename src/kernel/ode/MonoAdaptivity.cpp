// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005-11-04

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MonoAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptivity::MonoAdaptivity(const ODE& ode, const Method& method)
  : Adaptivity(ode, method), k(0)
{
  // Specify initial time step
  if ( kfixed )
  {
    k = ode.timestep(0.0);
  }
  else
  {
    k = dolfin_get("initial time step");
    if ( k > _kmax )
    {
      k = _kmax;
      dolfin_warning1("Initial time step larger than maximum time step, using k = %.3e.", k);
    }
  }

  // Initialize controller
  controller.init(k, safety*tol, method.order(), _kmax);
}
//-----------------------------------------------------------------------------
MonoAdaptivity::~MonoAdaptivity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real MonoAdaptivity::timestep() const
{
  return k;
}
//-----------------------------------------------------------------------------
void MonoAdaptivity::update(real k0, real r, const Method& method, real t)
{
  // Check if time step is fixed
  if ( kfixed )
  {
    k = ode.timestep(t);
    _accept = true;
    return;
  }

  // Compute local error estimate
  const real error = method.error(k0, r);
  
  // Let controller choose new time step
  k = controller.update(error, safety*tol);

  // Check if time step can be accepted
  _accept = true;    
  if ( error > tol )
  {
    k = std::min(k, 0.5*k0);
    controller.reset(k);
    _accept = false;

    //dolfin_info("e = %.3e  tol = %.3e", error, tol);
  }
}
//-----------------------------------------------------------------------------
