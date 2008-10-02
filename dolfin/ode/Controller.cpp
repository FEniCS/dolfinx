// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
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
Controller::Controller(double k, double tol, uint p, double kmax)
{
  init(k, tol, p, kmax);
}
//-----------------------------------------------------------------------------
Controller::~Controller()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Controller::init(double k, double tol, uint p, double kmax)
{
  k0 = k;
  k1 = k;
  e0 = tol;
  this->p = static_cast<double>(p);
  this->kmax = kmax;
}
//-----------------------------------------------------------------------------
void Controller::reset(double k)
{
  k0 = k;
  k1 = k;
}
//-----------------------------------------------------------------------------
double Controller::update(double e, double tol)
{
  return updateH211PI(e, tol);
}
//-----------------------------------------------------------------------------
double Controller::updateH0211(double e, double tol)
{
  // Compute new time step
  double k = k1*pow(tol/e, 1.0/(2.0*p))*pow(tol/e0, 1.0/(2.0*p))/sqrt(k1/k0);

  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);
  
  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;

  return k;
}
//-----------------------------------------------------------------------------
double Controller::updateH211PI(double e, double tol)
{
  // Compute new time step
  double k = k1*pow(tol/e, 1.0/(6.0*p))*pow(tol/e0, 1.0/(6.0*p));
    
  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;

  return k;
}
//-----------------------------------------------------------------------------
double Controller::updateSimple(double e, double tol)
{
  // Compute new time step
  double k = k1*pow(tol/e, 1.0/p);
  
  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;
  
  return k;
}
//-----------------------------------------------------------------------------
double Controller::updateHarmonic(double e, double tol)
{
  // Compute new time step
  double k = k1*pow(tol/e, 1.0/p);

  // Choose kmax if error is too small (should also catch nan or inf)
  if ( !(k <= kmax) )
    k = 2.0*k1*kmax / (k1 + kmax);

  // Take harmonic mean value with weight
  double w = 5.0;
  k = (1.0 + w)*k1*k / (k1 + w*k);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;

  return k;
}
//-----------------------------------------------------------------------------
double Controller::updateHarmonic(double knew, double kold, double kmax)
{
  const double w = 5.0;
  return std::min(kmax, (1.0 + w)*kold*knew / (kold + w*knew));
}
//-----------------------------------------------------------------------------
