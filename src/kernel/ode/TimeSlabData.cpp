// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabData::TimeSlabData(ODE& ode) : 
  components(ode.size()), regulators(ode.size())
{
  // Get parameters
  TOL                = dolfin_get("tolerance");
  kmax               = dolfin_get("maximum time step");
  interval_threshold = dolfin_get("interval threshold");
  _debug             = dolfin_get("debug time slab");
  real k0            = dolfin_get("initial time step");

  // Scale tolerance with the number of components
  TOL /= static_cast<real>(ode.size());

  // Get initial data
  for (unsigned int i = 0; i < components.size(); i++)
    components[i].u0 = ode.u0(i);
  
  // Specify initial time steps
  for (unsigned int i = 0; i < regulators.size(); i++)
    regulators[i].init(k0);

  // Open debug file
  if ( _debug )
    file.open("timeslab.debug", std::ios::out);
}
//-----------------------------------------------------------------------------
TimeSlabData::~TimeSlabData()
{
  // Close debug file
  if ( _debug )
    file.close();
}
//-----------------------------------------------------------------------------
Element* TimeSlabData::createElement(const Element::Type type, int q,
				     int index, TimeSlab* timeslab)
{
  //dolfin_debug3("creating element at: %d %lf-%lf", index,
  //		timeslab->starttime(), timeslab->endtime());

  // Create the new element
  Element *e = components[index].createElement(type, q, index, timeslab);

  //dolfin_debug2("components[%d].size(): %d", index, components[index].size());

  return e;
}
//-----------------------------------------------------------------------------
unsigned int TimeSlabData::size() const
{
  return components.size();
}
//-----------------------------------------------------------------------------
Component& TimeSlabData::component(unsigned int i)
{
  dolfin_assert(i < components.size());
  return components[i];
}
//-----------------------------------------------------------------------------
const Component& TimeSlabData::component(unsigned int i) const
{
  dolfin_assert(i < components.size());
  return components[i];
}
//-----------------------------------------------------------------------------
Regulator& TimeSlabData::regulator(unsigned int i)
{
  dolfin_assert(i < regulators.size());
  return regulators[i];
}
//-----------------------------------------------------------------------------
const Regulator& TimeSlabData::regulator(unsigned int i) const
{
  dolfin_assert(i < regulators.size());
  return regulators[i];
}
//-----------------------------------------------------------------------------
real TimeSlabData::tolerance() const
{
  return TOL;
}
//-----------------------------------------------------------------------------
real TimeSlabData::maxstep() const
{
  // FIXME: Should we have an individual kmax for each component?
  // In that case we should put kmax into the Regulator class.

  return kmax;
}
//-----------------------------------------------------------------------------
real TimeSlabData::threshold() const
{
  return interval_threshold;
}
//-----------------------------------------------------------------------------
void TimeSlabData::shift(TimeSlab& timeslab, RHS& f)
{
  // Specify new initial values for next time slab and compute
  // new time steps.

  for (unsigned int i = 0; i < components.size(); i++)
  {
    // Get last element
    Element& element = components[i].last();

    // Get value at the endpoint
    real u0 = element.endval();

    // Compute residual
    real r = element.computeResidual(f);

    // Compute new time step
    real k = element.computeTimeStep(TOL, r, kmax);

    // Update regulator
    regulators[i].update(k, kmax);

    // Clear component
    components[i].clear();

    // Specify new initial value
    components[i].u0 = u0;
  }
}
//-----------------------------------------------------------------------------
void TimeSlabData::debug(Element& element, Action action)
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
dolfin::LogStream& dolfin::operator<<(LogStream& stream, 
				      const TimeSlabData& data)
{
  stream << "[ TimeSlabData of size " << data.size() << " with components: "; 

  for (unsigned int i = 0; i < data.size(); i++)
    stream << endl << "  " << data.component(i);
  stream << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
