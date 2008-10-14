// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-29
// Last changed: 2008-02-11

#include <cmath>
#include <dolfin/parameter/parameters.h>
#include "ODE.h"
#include "Method.h"
#include "MonoAdaptivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptivity::MonoAdaptivity(const ODE& ode, const Method& method)
  : Adaptivity(ode, method), k(0)
{
  // Specify initial time step
  real k0 = static_cast<double>(ode.get("ODE initial time step"));
  if ( kfixed )
  {
    k = ode.timestep(0.0, k0);
  }
  else
  {
    k = k0;
    if ( k > _kmax )
    {
      k = _kmax;
      warning("Initial time step larger than maximum time step, using k = %.3e.", to_double(k));
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
void MonoAdaptivity::update(real k0, real r, const Method& method, real t,
			    bool first)
{
  // Check if time step is fixed
  if ( kfixed )
  {
    k = ode.timestep(t, k0);
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
    // Extra reduction if this is the first time step
    if ( first )
      k = min(k, 0.1*k0);
    else
      k = min(k, 0.5*k0);
    
    controller.reset(k);
    _accept = false;

    //message("e = %.3e  tol = %.3e", error, tol);
  }
}
//-----------------------------------------------------------------------------
