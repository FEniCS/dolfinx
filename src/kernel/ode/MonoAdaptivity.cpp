// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MonoAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptivity::MonoAdaptivity(ODE& ode)
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
  regulator.init(ode.timestep());
}
//-----------------------------------------------------------------------------
MonoAdaptivity::~MonoAdaptivity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real MonoAdaptivity::timestep() const
{
  return regulator.timestep();
}
//-----------------------------------------------------------------------------
void MonoAdaptivity::update(real r, const NewMethod& method)
{
  // Compute new time step
  const real k = method.timestep(r, tol, kmax_current);
  
  // Update regulator for component
  regulator.update(k, kmax_current, kfixed);
}
//-----------------------------------------------------------------------------
real MonoAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
