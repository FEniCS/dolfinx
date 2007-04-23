// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-11-02
// Last changed: 2005-11-03

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Controller.h>

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
  k0 = k;
  k1 = k;
  e0 = tol;
  this->p = static_cast<real>(p);
  this->kmax = kmax;
}
//-----------------------------------------------------------------------------
void Controller::reset(real k)
{
  k0 = k;
  k1 = k;
}
//-----------------------------------------------------------------------------
real Controller::update(real e, real tol)
{
  return updateH211PI(e, tol);
}
//-----------------------------------------------------------------------------
real Controller::updateH0211(real e, real tol)
{
  // Compute new time step
  real k = k1*pow(tol/e, 1.0/(2.0*p))*pow(tol/e0, 1.0/(2.0*p))/sqrt(k1/k0);

  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);
  
  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;

  return k;
}
//-----------------------------------------------------------------------------
real Controller::updateH211PI(real e, real tol)
{
  // Compute new time step
  real k = k1*pow(tol/e, 1.0/(6.0*p))*pow(tol/e0, 1.0/(6.0*p));
    
  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;

  return k;
}
//-----------------------------------------------------------------------------
real Controller::updateSimple(real e, real tol)
{
  // Compute new time step
  real k = k1*pow(tol/e, 1.0/p);
  
  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;
  
  return k;
}
//-----------------------------------------------------------------------------
real Controller::updateHarmonic(real e, real tol)
{
  // Compute new time step
  real k = k1*pow(tol/e, 1.0/p);

  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Take harmonic mean value with weight
  real w = 5.0;
  k = (1.0 + w)*k1*k / (k1 + w*k);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;

  return k;
}
//-----------------------------------------------------------------------------
real Controller::updateHarmonic(real knew, real kold, real kmax)
{
  const real w = 5.0;
  return std::min(kmax, (1.0 + w)*kold*knew / (kold + w*knew));
}
//-----------------------------------------------------------------------------
