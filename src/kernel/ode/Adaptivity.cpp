// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-04
// Last changed: 2005-11-04

#include <dolfin/dolfin_settings.h>
#include <dolfin/Adaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Adaptivity::Adaptivity(const ODE& ode, const Method& method)
  : ode(ode), method(method)
{
  tol    = dolfin_get("tolerance");
  _kmax  = dolfin_get("maximum time step");
  beta   = dolfin_get("interval threshold");
  safety = dolfin_get("safety factor");
  kfixed = dolfin_get("fixed time step");
   
  safety_old = safety;
  safety_max = safety;

  num_rejected = 0;

  _accept = true;

  // Scale tolerance with the square root of the number of components
  //tol /= sqrt(static_cast<real>(ode.size()));
}
//-----------------------------------------------------------------------------
Adaptivity::~Adaptivity()
{
  dolfin_info("Number of rejected time steps: %d", num_rejected);
}
//-----------------------------------------------------------------------------
bool Adaptivity::accept()
{
  if ( _accept )
  {
    safety_old = safety;
    safety = Controller::updateHarmonic(safety_max, safety_old, safety_max);
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
real Adaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
real Adaptivity::kmax() const
{
  return _kmax;
}
//-----------------------------------------------------------------------------
