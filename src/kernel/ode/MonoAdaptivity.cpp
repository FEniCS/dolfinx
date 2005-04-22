// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MonoAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptivity::MonoAdaptivity(ODE& ode) : k(0)
{
  // Get parameters
  tol    = dolfin_get("tolerance");
  kmax   = dolfin_get("maximum time step");
  kfixed = dolfin_get("fixed time step");
  beta   = dolfin_get("interval threshold");

  // Start with given maximum time step
  kmax_current = kmax;

  // Scale tolerance with the square root of the number of components
  //tol /= sqrt(static_cast<real>(ode.size()));

  // Specify initial time step
  k = ode.timestep();
  if ( k > kmax )
  {
    k = kmax;
    dolfin_warning1("Initial time step larger than maximum time step, using k = %.3e.", k);
  }
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
void MonoAdaptivity::update(real k0, real r, const NewMethod& method)
{
  // Compute new time step
  const real k1 = method.timestep(r, tol, k0, kmax_current);
  
  // Regulate the time step
  //k = regulator.regulate(k1, k0, kmax_current, kfixed);
  k = regulator.regulate(k1, k, kmax_current, kfixed);
}
//-----------------------------------------------------------------------------
real MonoAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
