// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Element.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>
#include <dolfin/Adaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Adaptivity::Adaptivity(ODE& ode) : regulators(ode.size())
{
  // Get parameters
  TOL     = dolfin_get("tolerance");
  kmax    = dolfin_get("maximum time step");
  kfixed  = dolfin_get("fixed time step");
  beta    = dolfin_get("interval threshold");

  // Start with given maximum time step
  kmax_current = kmax;

  // Scale tolerance with the square root of the number of components
  //TOL /= sqrt(static_cast<real>(ode.size()));

  // Specify initial time steps
  for (unsigned int i = 0; i < regulators.size(); i++)
    regulators[i].init(ode.timestep(i));

  // Remaining number of small time steps
  m = 0;
}
//-----------------------------------------------------------------------------
Adaptivity::~Adaptivity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Regulator& Adaptivity::regulator(unsigned int i)
{
  dolfin_assert(i < regulators.size());
  return regulators[i];
}
//-----------------------------------------------------------------------------
const Regulator& Adaptivity::regulator(unsigned int i) const
{
  dolfin_assert(i < regulators.size());
  return regulators[i];
}
//-----------------------------------------------------------------------------
real Adaptivity::tolerance() const
{
  return TOL;
}
//-----------------------------------------------------------------------------
real Adaptivity::maxstep() const
{
  return kmax_current;
}
//-----------------------------------------------------------------------------
real Adaptivity::minstep() const
{
  real kmin = regulators[0].timestep();
  for (unsigned int i = 1; i < regulators.size(); i++)
    kmin = std::min(kmin, regulators[i].timestep());
  
  return std::min(kmin, kmax_current);
}
//-----------------------------------------------------------------------------
void Adaptivity::stabilize(real k, unsigned int m)
{
  dolfin_assert(k >= 0.0);
  dolfin_assert(m >= 1);

  cout << "Stabilizing time step" << endl;

  // Decrease time step by at least a factor 1/2
  kmax_current = k;
  this->m = m;

  // Update regulators
  for (unsigned int i = 0; i < regulators.size(); i++)
    regulators[i].update(kmax_current);
}
//-----------------------------------------------------------------------------
bool Adaptivity::fixed() const
{
  return kfixed;
}
//-----------------------------------------------------------------------------
real Adaptivity::threshold() const
{
  return beta;
}
//-----------------------------------------------------------------------------
unsigned int Adaptivity::size() const
{
  return regulators.size();
}
//-----------------------------------------------------------------------------
void Adaptivity::shift(Solution& u, RHS& f)
{
  // Update residuals and time steps
  for (unsigned int i = 0; i < u.size(); i++)
  {
    // Get last element
    Element* element = u.last(i);
    dolfin_assert(element);

    // Compute residual
    real r = element->computeResidual(f);

    // Compute new time step
    real k = element->computeTimeStep(TOL, r, kmax_current);

    // Update regulator
    regulators[i].update(k, kmax_current, kfixed);
  }
  
  // Update stabilization
  if ( m > 0 )
  {
    // Decrease the remaining number of small steps
    m -= 1;
  }
  else

  {
    // Increase kmax_current with a factor 2 towards kmax
    kmax_current = 2.0 * kmax_current * kmax / (kmax_current + kmax);
  }
}
//-----------------------------------------------------------------------------
bool Adaptivity::accept(TimeSlab& timeslab, RHS& f)
{
  // FIXME: Use one margin for determining the time steps and another
  // FIXME: margin for accepting time slabs. Make these parameters.
  return timeslab.accept(f, 2*TOL);
}
//-----------------------------------------------------------------------------
void Adaptivity::adjustMaximumTimeStep(real kmax)
{
  this->kmax = kmax;
  kmax_current = kmax;
  
  cout << "Adjusting maximum time step to " << kmax << endl;
}
//-----------------------------------------------------------------------------
