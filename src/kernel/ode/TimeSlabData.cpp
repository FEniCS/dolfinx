// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabData::TimeSlabData(ODE& ode) : 
  components(ode.size()), regulators(ode.size())
{
  // Get parameters
  _debug = dolfin_get("debug time slab");
  real k = dolfin_get("initial time step");
  interval_threshold = dolfin_get("interval threshold");

  // Get initial data
  for (unsigned int i = 0; i < components.size(); i++)
    components[i].u0 = ode.u0(i);
  
  // Specify initial time steps
  for (unsigned int i = 0; i < regulators.size(); i++)
    regulators[i].init(k);

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
real TimeSlabData::threshold() const
{
  return interval_threshold;
}
//-----------------------------------------------------------------------------
void TimeSlabData::shift(TimeSlab& timeslab)
{
  /// Clear data for the current interval and shift values to the next interval

  //dolfin_debug("foo");

  typedef std::vector<Component>::iterator dataiterator;
  for (dataiterator it = components.begin(); it != components.end(); it++)
  {
    Component &c = *it;
    
    //dolfin_debug1("c.size(): %d", c.size());

    real u0i = c.last().eval(timeslab.endtime());
    //real u0i = c(topslab->endtime());
    
    //dolfin_debug("--- Calling c = Component()");
    c = Component();
    //dolfin_debug("--- Done");

    c.u0 = u0i;

    //dolfin::cout << "Last element at: " << c.last().starttime() << "-" <<
    //c.last().endtime() << dolfin::endl;
    //dolfin_debug2("u0i: %d, %lf", it - components.begin(), u0i);

  }
}
//-----------------------------------------------------------------------------
void TimeSlabData::debug(Element& element, Action action)
{
  if ( !_debug )
    return;

  //Write debug info to file
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
