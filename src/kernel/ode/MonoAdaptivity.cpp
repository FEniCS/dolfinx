// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005-11-01

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MonoAdaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptivity::MonoAdaptivity(ODE& ode, const Method& method)
  : k(0), _accept(false), num_rejected(0)
{
  // Get parameters
  tol    = dolfin_get("tolerance");
  kmax   = dolfin_get("maximum time step");
  kfixed = dolfin_get("fixed time step");
  beta   = dolfin_get("interval threshold");
  safety = dolfin_get("safety factor");
  
  // Start with given maximum time step
  kmax_current = kmax;

  // Start with maximum allowed safety factor
  safety_max = safety;
  safety_old = safety;

  // Scale tolerance with the square root of the number of components
  //tol /= sqrt(static_cast<real>(ode.size()));

  // Specify initial time step
  k = ode.timestep();
  if ( k > kmax )
  {
    k = kmax;
    dolfin_warning1("Initial time step larger than maximum time step, using k = %.3e.", k);
  }

  // Initialize controller
  controller.init(k, safety*tol, method.order());
}
//-----------------------------------------------------------------------------
MonoAdaptivity::~MonoAdaptivity()
{
  dolfin_info("Number of rejected time steps: %d", num_rejected);
}
//-----------------------------------------------------------------------------
real MonoAdaptivity::timestep() const
{
  return k;
}
//-----------------------------------------------------------------------------
void MonoAdaptivity::update(real k0, real r, const Method& method)
{
  // Check if time step is fixed
  if ( kfixed )
  {
    _accept = true;
    return;
  }

  // Compute new time step
  //const real k1 = method.timestep(r, safety*tol, k0, kmax_current);

  // Regulate the time step
  //k = regulator.regulate(k1, k0, kmax_current, kfixed);
  //k = regulator.regulate(k1, k, kmax_current, kfixed);
  

  //k = (1.0 + 20.0)*k1*k/(k+20.0*k1);

  //k = 2.0*k1*k0/(k0+k1);
  
  // Compute local error estimate
  const real error = method.error(k0, r);
  
  //k = controller.updateSimple(error, safety*tol);
  //k = controller.updateHarmonic(error, safety*tol);
  //k = controller.updateH0211(error, safety*tol);
  k = controller.updateH211PI(error, safety*tol);

  _accept = true;  
  
  if ( error > tol )
  {
    k = std::min(k, 0.5*k0);
    _accept = false;

    //dolfin_info("e = %.3e  tol = %.3e", error, tol);
    //safety = regulator.regulate(tol/error, safety, 1.0, false);
  }
}
//-----------------------------------------------------------------------------
bool MonoAdaptivity::accept()
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
      safety_max = safety_old;
      safety_old = safety;
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
real MonoAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
