// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005-11-01

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptivity::MultiAdaptivity(ODE& ode)
  : regulators(ode.size()), timesteps(0), _accept(false), num_rejected(0)
{
  // Get parameters
  tol    = dolfin_get("tolerance");
  kmax   = dolfin_get("maximum time step");
  kfixed = dolfin_get("fixed time step");
  beta   = dolfin_get("interval threshold");
  safety = dolfin_get("safety factor");
  
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
  dolfin_info("Number of rejected time steps: %d", num_rejected);

  if ( timesteps ) delete [] timesteps;
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::timestep(uint i) const
{
  return timesteps[i];
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::updateInit()
{
  // Will remain true if solution can be accepted for all components
  _accept = true;
}
//-----------------------------------------------------------------------------
void MultiAdaptivity::updateComponent(uint i, real k0, real r,
				      const Method& method)
{
  // Compute new time step
  real k1 = method.timestep(r, safety*tol, k0, kmax_current);
  
  // Regulate the time step
  timesteps[i] = regulator.regulate(k1, k0, kmax_current, kfixed);

  // Check the size of the residual
  const real error = method.error(k0, r);
  if ( error > tol )
  {
    //dolfin_info("i = %d e = %.3e  tol = %.3e", i, error, tol);
    safety = std::min(safety, regulator.regulate(tol/error, safety, 1.0, false));
    _accept = false;
  }

  /*
  cout << "i = " << i << endl;
  cout << "Residual:  " << fabs(r) << endl;
  cout << "Previous:  " << k0 << endl;
  cout << "Suggested: " << k1 << endl;
  cout << "Regulated: " << timesteps[i] << endl << endl;
  */
}
//-----------------------------------------------------------------------------
bool MultiAdaptivity::accept()
{
  if ( !_accept )
    num_rejected++;
  
  return _accept;
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
