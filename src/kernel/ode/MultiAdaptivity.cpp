// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MultiAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptivity::MultiAdaptivity(ODE& ode) : regulators(ode.size())
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

  // Specify initial time steps
  for (uint i = 0; i < regulators.size(); i++)
    regulators[i].init(ode.timestep(i));
}
//-----------------------------------------------------------------------------
MultiAdaptivity::~MultiAdaptivity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::timestep(uint i) const
{
  return regulators[i].timestep();
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::update(uint i, real r, const NewMethod& method)
{
  // Compute new time step
  const real k = method.timestep(r, tol, kmax_current);
  
  // Update regulator for component
  regulators[i].update(k, kmax_current, kfixed);
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
