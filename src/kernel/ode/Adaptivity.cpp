// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Element.h>
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

  // Scale tolerance with the number of components
  TOL /= static_cast<real>(ode.size());

  // Specify initial time steps
  for (unsigned int i = 0; i < regulators.size(); i++)
    regulators[i].init(ode.timestep(i));
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
void Adaptivity::decreaseTimeStep(real factor)
{
  dolfin_assert(factor >= 0.0);
  dolfin_assert(factor <= 1.0);
  kmax_current = factor * minstep();
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
void Adaptivity::shift()
{
  // FIXME: Maybe this should be a parameter
  kmax_current *= 1.5;
  if ( kmax_current > kmax )
    kmax_current = kmax;
}
//-----------------------------------------------------------------------------
