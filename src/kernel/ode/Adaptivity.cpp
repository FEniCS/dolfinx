// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/Element.h>
#include <dolfin/Adaptivity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Adaptivity::Adaptivity(unsigned int N) : regulators(N)
{
  // Get parameters
  TOL     = dolfin_get("tolerance");
  kmax    = dolfin_get("maximum time step");
  beta    = dolfin_get("interval threshold");
  real k0 = dolfin_get("initial time step");

  // Scale tolerance with the number of components
  TOL /= static_cast<real>(N);

  // Specify initial time steps
  for (unsigned int i = 0; i < regulators.size(); i++)
    regulators[i].init(k0);
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
  // FIXME: Should we have an individual kmax for each component?
  // FIXME: In that case we should put kmax into the Regulator class.

  return kmax;
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
