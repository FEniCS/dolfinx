// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/RHS.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSteppingData.h>
#include <dolfin/ElementData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementData::ElementData(const ODE& ode) : components(ode.size())
{
  // Get initial data
  for (unsigned int i = 0; i < components.size(); i++)
    components[i].setu0(ode.u0(i));
}
//-----------------------------------------------------------------------------
ElementData::~ElementData()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Element* ElementData::createElement(const Element::Type type, int q,
				     int index, TimeSlab* timeslab)
{
  // Create the new element
  Element* e = components[index].createElement(type, q, index, timeslab);

  return e;
}
//-----------------------------------------------------------------------------
unsigned int ElementData::size() const
{
  return components.size();
}
//-----------------------------------------------------------------------------
Component& ElementData::component(unsigned int i)
{
  dolfin_assert(i < components.size());
  return components[i];
}
//-----------------------------------------------------------------------------
const Component& ElementData::component(unsigned int i) const
{
  dolfin_assert(i < components.size());
  return components[i];
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, 
				      const ElementData& data)
{
  stream << "[ ElementData of size " << data.size() << " with components: "; 

  for (unsigned int i = 0; i < data.size(); i++)
    stream << endl << "  " << data.component(i);
  stream << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
