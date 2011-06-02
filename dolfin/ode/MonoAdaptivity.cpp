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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-29
// Last changed: 2010-04-05

#include <cmath>
#include "ODE.h"
#include "Method.h"
#include "MonoAdaptivity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptivity::MonoAdaptivity(const ODE& ode, const Method& method)
  : Adaptivity(ode, method), k(0)
{
  // Specify initial time step
  real k0 = ode.parameters["initial_time_step"].get_real();
  if ( kfixed )
  {
    k = ode.timestep(0.0, k0);
  }
  else
  {
    k = k0;
    if ( k > _kmax )
    {
      k = _kmax;
      warning("Initial time step larger than maximum time step, using k = %.3e.", to_double(k));
    }
  }

  // Initialize controller
  controller.init(k, safety*tol, method.order(), _kmax);
}
//-----------------------------------------------------------------------------
MonoAdaptivity::~MonoAdaptivity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real MonoAdaptivity::timestep() const
{
  return k;
}
//-----------------------------------------------------------------------------
void MonoAdaptivity::update(real k0, real r, const Method& method, real t,
			    bool first)
{
  // Check if time step is fixed
  if ( kfixed )
  {
    k = ode.timestep(t, k0);
    _accept = true;
    return;
  }

  // Compute local error estimate
  const real error = method.error(k0, r);

  // Let controller choose new time step
  k = controller.update(error, safety*tol);

  // Check if time step can be accepted
  _accept = true;
  if ( error > tol )
  {
    // Extra reduction if this is the first time step
    if ( first )
      k = real_min(k, 0.1*k0);
    else
      k = real_min(k, 0.5*k0);

    controller.reset(k);
    _accept = false;

    //info(DBG, "e = %.3e  tol = %.3e", error, tol);
  }
}
//-----------------------------------------------------------------------------
