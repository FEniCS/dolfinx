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
MonoAdaptivity::MonoAdaptivity(ODE& ode)
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

  // Start with 

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
  // Compute new time step
  const real k1 = method.timestep(r, safety*tol, k0, kmax_current);
  
  // Regulate the time step
  //k = regulator.regulate(k1, k0, kmax_current, kfixed);
  k = regulator.regulate(k1, k, kmax_current, kfixed);

  // Check the size of the residual
  const real error = method.error(k0, r);
  //dolfin_info("e = %.3e  tol = %.3e", error, tol);
  _accept = error <= tol;
}
//-----------------------------------------------------------------------------
bool MonoAdaptivity::accept()
{
  if ( !_accept )
    num_rejected++;
  
  return _accept;
}
//-----------------------------------------------------------------------------
real MonoAdaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
