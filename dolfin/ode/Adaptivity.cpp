// Copyright (C) 2005-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-11-04
// Last changed: 2009-09-08

#include "Adaptivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Adaptivity::Adaptivity(const ODE& ode, const Method& method)
  : ode(ode), method(method)
{
  tol    = ode.parameters["tolerance"].get_real();
  _kmax  = ode.parameters["maximum_time_step"].get_real();
  beta   = ode.parameters["interval_threshold"].get_real();
  safety = ode.parameters["safety_factor"].get_real();
  kfixed = ode.parameters["fixed_time_step"];

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
  info("Number of rejected time steps: %d", num_rejected);
}
//-----------------------------------------------------------------------------
bool Adaptivity::accept()
{
  if ( _accept )
  {
    safety_old = safety;
    safety = Controller::update_harmonic(safety_max, safety_old, safety_max);
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

  //info("safefy factor = %.3e", safety);

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
