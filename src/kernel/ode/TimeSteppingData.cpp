// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Element.h>
#include <dolfin/Component.h>
#include <dolfin/ElementData.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSteppingData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSteppingData::TimeSteppingData(ODE& ode, ElementData& elmdata, real t0) : 
  elmdata(elmdata), regulators(ode.size()), initval(ode.size()), t0(t0)
{
  // Get parameters
  TOL                = dolfin_get("tolerance");
  kmax               = dolfin_get("maximum time step");
  interval_threshold = dolfin_get("interval threshold");
  _debug             = dolfin_get("debug time steps");
  real k0            = dolfin_get("initial time step");

  // Scale tolerance with the number of components
  TOL /= static_cast<real>(elmdata.size());

  // Specify initial time steps
  for (unsigned int i = 0; i < regulators.size(); i++)
    regulators[i].init(k0);

  // Set initial data
  for (unsigned int i = 0; i < initval.size(); i++)
    initval[i] = ode.u0(i);

  // Open debug file
  if ( _debug )
    file.open("timesteps.debug", std::ios::out);
}
//-----------------------------------------------------------------------------
TimeSteppingData::~TimeSteppingData()
{
  // Close debug file
  if ( _debug )
    file.close();
}
//-----------------------------------------------------------------------------
Element* TimeSteppingData::createElement(Element::Type type, real t0, real t1,
					 int q, int index)
{
  return elmdata.createElement(type, t0, t1, q, index);
}
//-----------------------------------------------------------------------------
Element* TimeSteppingData::element(unsigned int i, real t)
{
  return elmdata.element(i, t);
}
//-----------------------------------------------------------------------------
unsigned int TimeSteppingData::size() const
{
  return elmdata.size();
}
//-----------------------------------------------------------------------------
Regulator& TimeSteppingData::regulator(unsigned int i)
{
  dolfin_assert(i < regulators.size());
  return regulators[i];
}
//-----------------------------------------------------------------------------
const Regulator& TimeSteppingData::regulator(unsigned int i) const
{
  dolfin_assert(i < regulators.size());
  return regulators[i];
}
//-----------------------------------------------------------------------------
real TimeSteppingData::u(unsigned int i, real t) const
{
  dolfin_assert(i < initval.size());

  // Note: the logic of this function is nontrivial...

  // Return initial value at t0
  if ( t == t0 )
    return initval[i];

  // Try to find the element
  Element* element = elmdata.element(i,t);

  // Return value from element if we found it
  if ( element )
    return element->value(t);

  // Otherwise, return the initial value
  return initval[i];
}
//-----------------------------------------------------------------------------
real TimeSteppingData::k(unsigned int i, real t) const
{
  Element* element = elmdata.element(i,t);
  dolfin_assert(element);

  return element->timestep();
}
//-----------------------------------------------------------------------------
real TimeSteppingData::r(unsigned int i, real t, RHS& f) const
{
  Element* element = elmdata.element(i,t);
  dolfin_assert(element);

  return element->computeResidual(f);
}
//-----------------------------------------------------------------------------
real TimeSteppingData::tolerance() const
{
  return TOL;
}
//-----------------------------------------------------------------------------
real TimeSteppingData::maxstep() const
{
  // FIXME: Should we have an individual kmax for each component?
  // FIXME: In that case we should put kmax into the Regulator class.

  return kmax;
}
//-----------------------------------------------------------------------------
real TimeSteppingData::threshold() const
{
  return interval_threshold;
}
//-----------------------------------------------------------------------------
void TimeSteppingData::shift(TimeSlab& timeslab, RHS& f)
{
  // Specify new initial values for next time slab and compute
  // new time steps.

  for (unsigned int i = 0; i < elmdata.size(); i++)
  {
    // Get last element
    Element* element = elmdata.last(i);
    dolfin_assert(element);

    // Compute residual
    real r = element->computeResidual(f);

    // Compute new time step
    real k = element->computeTimeStep(TOL, r, kmax);

    // Update regulator
    regulators[i].update(k, kmax);

    // Update initial value
    initval[i] = element->endval();
  }
  
  // Clear element data
  elmdata.clear();

  // Update time for initial values
  t0 = timeslab.endtime();
}
//-----------------------------------------------------------------------------
void TimeSteppingData::debug(Element& element, Action action)
{
  if ( !_debug )
    return;

  // Write debug info to file
  file << action << " "
       << element.index() << " " 
       << element.starttime() << " " 
       << element.endtime() << "\n";
}
//-----------------------------------------------------------------------------
