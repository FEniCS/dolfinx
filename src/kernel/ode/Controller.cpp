// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-02
// Last changed: 2005-11-02

#include <cmath>
#include <dolfin/Controller.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Controller::Controller()
{
  init(0.0, 0.0, 0);
}
//-----------------------------------------------------------------------------
Controller::Controller(real k, real tol, uint p)
{
  init(k, tol, p);
}
//-----------------------------------------------------------------------------
Controller::~Controller()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Controller::init(real k, real tol, uint p)
{
  k0 = k;
  k1 = k;
  e0 = tol;
  this->p = static_cast<real>(p);
}
//-----------------------------------------------------------------------------
real Controller::update(real e, real tol)
{
  return updateH0211(e, tol);
}
//-----------------------------------------------------------------------------
real Controller::updateH0211(real e, real tol)
{
  // Compute new time step
  real k = k1*pow(tol/e, 1.0/(2.0*p))*pow(tol/e0, 1.0/(2.0*p))/sqrt(k1/k0);
  
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

  // Take harmonic mean value with weight
  real w = 5.0;
  k = (1.0 + w) * k1 * k / (k1 + w*k);

  // Update history (note that e1 == e)
  k0 = k1; k1 = k;
  e0 = e;

  return k;
}
//-----------------------------------------------------------------------------
