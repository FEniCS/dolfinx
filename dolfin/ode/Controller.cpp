// Copyright (C) 2005 Anders Logg
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
// First added:  2005-11-02
// Last changed: 2005-11-03

#include <cmath>
#include <dolfin/log/dolfin_log.h>
#include "Controller.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Controller::Controller()
{
  init(0.0, 0.0, 0, 0.0);
}
//-----------------------------------------------------------------------------
Controller::Controller(real k, real tol, uint p, real kmax)
{
  init(k, tol, p, kmax);
}
//-----------------------------------------------------------------------------
Controller::~Controller()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Controller::init(real k, real tol, uint p, real kmax)
{
  k0 = to_double(k);
  k1 = to_double(k);
  e0 = to_double(tol);
  this->p = static_cast<double>(p);
  this->kmax = to_double(kmax);
}
//-----------------------------------------------------------------------------
void Controller::reset(real k)
{
  k0 = to_double(k);
  k1 = to_double(k);
}
//-----------------------------------------------------------------------------
real Controller::update(real e, real tol)
{
  return updateH211PI(e, tol);
}
//-----------------------------------------------------------------------------
real Controller::updateH0211(real e, real tol)
{
  double _e   = to_double(e);
  double _tol = to_double(tol);

  // Compute new time step
  double k = k1*std::pow(_tol/_e, 1.0/(2.0*p))*std::pow(_tol/e0, 1.0/(2.0*p))/std::sqrt(k1/k0);

  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = _e;

  return k;
}
//-----------------------------------------------------------------------------
real Controller::updateH211PI(real e, real tol)
{
  double _e   = to_double(e);
  double _tol = to_double(tol);


  // Compute new time step
  double k = k1*std::pow(_tol/_e, 1.0/(6.0*p))*std::pow(_tol/e0, 1.0/(6.0*p));

  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = _e;

  return k;
}
//-----------------------------------------------------------------------------
real Controller::update_simple(real e, real tol)
{
  double _e   = to_double(e);
  double _tol = to_double(tol);

  // Compute new time step
  double k = k1*std::pow(_tol/_e, 1.0/p);

  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = _e;

  return k;
}
//-----------------------------------------------------------------------------
real Controller::update_harmonic(real e, real tol)
{
  double _e   = to_double(e);
  double _tol = to_double(tol);


  // Compute new time step
  double k = k1*std::pow(_tol/_e, 1.0/p);

  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Take harmonic mean value with weight
  double w = 5.0;
  k = (1.0 + w)*k1*k / (k1 + w*k);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = _e;

  return k;
}
//-----------------------------------------------------------------------------
real Controller::update_harmonic(real knew, real kold, real kmax)
{
  const real w = 5.0;
  return real_min(kmax, (1.0 + w)*kold*knew / (kold + w*knew));
}
//-----------------------------------------------------------------------------
