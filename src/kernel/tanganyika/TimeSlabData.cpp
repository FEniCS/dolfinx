// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabData::TimeSlabData(ODE& ode)
{
  topslab = 0;
  components.init(ode.size());
}
//-----------------------------------------------------------------------------
TimeSlabData::~TimeSlabData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Table<Element>::Iterator TimeSlabData::createElement(Element::Type type, int q,
						    int index, TimeSlab* timeslab)
{
  // Create element
  int id;
  Element* element = elements.create(&id);

  // Add element to the component list
  int pos = components(index).add(*element, topslab->endtime());

  // Initialize element
  element->init(type, q, index, pos, timeslab);

  // Return an iterator positioned at the element
  return elements.iterator(id);
}
//-----------------------------------------------------------------------------
void TimeSlabData::setslab(TimeSlab* timeslab)
{
  topslab = timeslab;
}
//-----------------------------------------------------------------------------
int TimeSlabData::size() const
{
  return elements.size();
}
//-----------------------------------------------------------------------------
Component& TimeSlabData::component(int i)
{
  return components(i);
}
//-----------------------------------------------------------------------------
