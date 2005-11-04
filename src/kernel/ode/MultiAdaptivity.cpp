// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005-11-02

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptivity::MultiAdaptivity(ODE& ode, const Method& method)
  : regulators(ode.size()), timesteps(0), controllers(ode.size()),
    _accept(false), num_rejected(0)
{
  // Get parameters
  tol    = dolfin_get("tolerance");
  kmax   = dolfin_get("maximum time step");
  kfixed = dolfin_get("fixed time step");
  beta   = dolfin_get("interval threshold");
  safety = dolfin_get("safety factor");
  weight = dolfin_get("time step conservation");
  
  // Initialize time steps
  timesteps = new real[ode.size()];
  for (uint i = 0; i < ode.size(); i++)
    timesteps[i] = 0.0;

  // Start with given maximum time step
  kmax_current = kmax;

  // Start with maximum allowed safety factor
  safety_max = safety;
  safety_old = safety;

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

    controllers[i].init(k, safety*tol, method.order(), kmax);
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
  // Check if time step is fixed
  if ( kfixed )
  {
    _accept = true;
    return;
  }

  // Compute local error estimate
  const real error = method.error(k0, r);
  
  // Let controller choose new time step
  //real k = controllers[i].update(error, safety*tol);


  //real k = controllers[i].updateHarmonic(error, safety*tol, timesteps[i]);
  //real k1 = method.timestep(r, safety*tol, k0, kmax_current);
  //real k = regulator.regulate(k1, timesteps[i], kmax_current, kfixed);

  real k1 = timesteps[i];
  real k  = method.timestep(r, safety*tol, k0, kmax_current);
  k = std::min(kmax_current, (1.0 + weight) * k1 * k / (k1 + weight*k));

  //dolfin_info("i = %d k0 = %.3e k_old = %.3e k_new = %.3e r = %.3e e = %.3e", 
  //	i, k0, k1, k, r, error);
  //  k = k1;
  
  // Check if time step can be accepted
  if ( error > tol )
  {
    k = std::min(k, 0.5*k0);
    _accept = false;
    
    // FIXME: Use method.order()
    controllers[i].init(tol, safety*tol, 2, kmax);

    //dolfin_info("e = %.3e  tol = %.3e", error, tol);
    //safety = regulator.regulate(tol/error, safety, 1.0, false);
  }

  /*
  //  if ( i == 0 )
  // {
  cout << "i = " << i << endl;
  cout << "Residual:  " << fabs(r) << endl;
  cout << "Previous:  " << timesteps[i] << endl;
  cout << "Used:      " << k0 << endl;
  cout << "Suggested: " << k  << endl;
  //}
  */

  // Save time step for component
  timesteps[i] = k;
}
//-----------------------------------------------------------------------------
bool MultiAdaptivity::accept()
{
  if ( _accept )
  {
    safety_old = safety;
    safety = regulator.regulate(safety_max, safety_old, 1.0, false);
    //cout << "---------------------- Time step ok -----------------------" << endl;
  }
  else
  {
    if ( safety > safety_old )
    {
      safety_max = 0.9*safety_old;
      safety_old = safety;
      safety = safety_max;
    }
    else
    {
      safety_old = safety;
      safety = 0.5*safety;
    }

    num_rejected++;
  }

  //dolfin_info("safefy factor = %.3e", safety);
  
  return _accept;
}
//-----------------------------------------------------------------------------
real MultiAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
