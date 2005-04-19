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
  w      = dolfin_get("time step conservation");
  margin = 1.0;
  
  // Start with given maximum time step
  kmax_current = kmax;
  
  // Scale tolerance with the square root of the number of components
  //tol /= sqrt(static_cast<real>(ode.size()));
  
  // Specify initial time steps
  bool modified = false;
  for (uint i = 0; i < regulators.size(); i++)
  {
    real k = ode.timestep(i);
    if ( k > kmax )
    {
      k = kmax;
      modified = true;
    }
    regulators[i].init(k);
  }
  if ( modified )
    dolfin_warning1("Initial time step too large for at least one component, using k = %.3e.", kmax);
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
  // Update margin
  //const real error = method.error(regulators[i].timestep(), r);
  //if ( error > tol )
  //  margin = tol / error;
  //cout << "margin = " << margin << endl;

  // Compute new time step
  const real k = method.timestep(r, tol*margin, kmax_current);
  
  // Update regulator for component
  regulators[i].update(k, kmax_current, w, kfixed);
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
