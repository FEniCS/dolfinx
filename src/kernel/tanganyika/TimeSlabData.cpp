// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabData::TimeSlabData(ODE& ode) : components(ode.size())
{
  /*
  topslab = 0;
  components.init(ode.size());
  //dolfin_debug1("components(index): %p", &(components(0)));
  //dolfin_debug1("components(index): %p", &(components(1)));
  */

}
//-----------------------------------------------------------------------------
TimeSlabData::~TimeSlabData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Element TimeSlabData::createElement(Element::Type type, int q, int index, TimeSlab* timeslab)
{
  Element element(type, q, index, timeslab);

  // Add element to the component list
  components[index].add(element, timeslab->endtime());

  return element;
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
