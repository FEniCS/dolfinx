// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabData::TimeSlabData(ODE& ode) : components(ode.size())
{
  topslab = 0;
  
  for(int i = 0; i < components.size(); i++)
  {
    components[i].u0 = ode.u0(i);
  }

  //dolfin_debug1("components(index): %p", &(components(0)));
  //dolfin_debug1("components(index): %p", &(components(1)));
}
//-----------------------------------------------------------------------------
TimeSlabData::~TimeSlabData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Element* TimeSlabData::createElement(const Element::Type type, int q,
				     int index, TimeSlab* timeslab)
{
  dolfin_debug3("creating element at: %d %lf-%lf", index,
		timeslab->starttime(), timeslab->endtime());

  Element *e = components[index].createElement(type, q, index, timeslab);

  dolfin_debug2("components[%d].size(): %d", index, components[index].size());

  return e;
}
//-----------------------------------------------------------------------------
void TimeSlabData::setslab(TimeSlab* timeslab)
{
  topslab = timeslab;
}
//-----------------------------------------------------------------------------
int TimeSlabData::size() const
{
  return components.size();
}
//-----------------------------------------------------------------------------
Component& TimeSlabData::component(int i)
{
  return components[i];
}
//-----------------------------------------------------------------------------
void TimeSlabData::shift()
{
  /// Clear data for the current interval and shift values to the next interval

  dolfin_debug("foo");

  typedef std::vector<Component>::iterator dataiterator;
  for(dataiterator it = components.begin(); it != components.end(); it++)
  {
    Component &c = *it;
    
    //dolfin_debug1("c.size(): %d", c.size());

    real u0i = c.last().eval(topslab->endtime());
    //real u0i = c(topslab->endtime());
    c = Component();
    c.u0 = u0i;

    //dolfin::cout << "Last element at: " << c.last().starttime() << "-" <<
    //c.last().endtime() << dolfin::endl;
    dolfin_debug2("u0i: %d, %lf", it - components.begin(), u0i);

  }
}
//-----------------------------------------------------------------------------
