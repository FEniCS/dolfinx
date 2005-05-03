// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptivity::MultiAdaptivity(ODE& ode) : regulators(ode.size()), timesteps(0)
{
  // Get parameters
  tol    = dolfin_get("tolerance");
  kmax   = dolfin_get("maximum time step");
  kfixed = dolfin_get("fixed time step");
  beta   = dolfin_get("interval threshold");
  
  // Initialize time steps
  timesteps = new real[ode.size()];
  for (uint i = 0; i < ode.size(); i++)
    timesteps[i] = 0.0;

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
    timesteps[i] = k;
  }
  if ( modified )
    dolfin_warning1("Initial time step too large for at least one component, using k = %.3e.", kmax);
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
void MultiAdaptivity::update(uint i, real k0, real r, const Method& method)
{
  // Compute new time step
  const real k1 = method.timestep(r, tol, k0, kmax_current);
  
  // Regulate the time step
  //timesteps[i] = regulator.regulate(k1, k0, kmax_current, kfixed);
  timesteps[i] = regulator.regulate(k1, timesteps[i], kmax_current, kfixed);

  /*
  cout << "i = " << i << endl;
  cout << "Residual:  " << fabs(r) << endl;
  cout << "Previous:  " << k0 << endl;
  cout << "Suggested: " << k1 << endl;
  cout << "Regulated: " << timesteps[i] << endl << endl;
  */

  /*
  if ( i == 145 )
  {
    cout << "Old step:  " << k0 << endl;
    cout << "Suggested: " << k1 << endl;
    cout << "Regulated: " << timesteps[i] << endl;
  }
  */
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
