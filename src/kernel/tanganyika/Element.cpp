// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/cGqElement.h>
#include <dolfin/dGqElement.h>
#include <dolfin/GenericElement.h>
#include <dolfin/Element.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Element::Element()
{
  element = 0;
}
//-----------------------------------------------------------------------------
Element::Element(Type type, int q, int index, TimeSlab* timeslab)
{
  switch (type) {
  case cg:
    element = new cGqElement(q, index, timeslab);
    break;
  case dg:
    element = new dGqElement(q, index, timeslab);
    break;
  default:
    dolfin_error("Unknown element type");
    element = 0;
  }
}
//-----------------------------------------------------------------------------
Element::~Element()
{
  dolfin_debug("foo");

  //if ( element )
  //delete element;
  //element = 0;
}
/*
//-----------------------------------------------------------------------------
void Element::init(Type type, int q, int index, TimeSlab* timeslab)
{
  dolfin_debug1("Created element component: %d", index);

  if ( element ) {
    dolfin_debug("FIXME: don't delete the old element");
    delete element;
  }

  dolfin_debug1("slab: %p", timeslab);

  switch (type) {
  case cg:
    element = new cGqElement(q, index, timeslab);
    dolfin_debug1("order: %d", element->order());
    break;
  case dg:
    element = new dGqElement(q, index, timeslab);
    break;
  default:
    dolfin_error("Unknown element type");
    element = 0;
  }
}
//-----------------------------------------------------------------------------
*/
real Element::eval(real t) const
{
  dolfin_assert(element);
  return element->eval(t);
}
//-----------------------------------------------------------------------------
real Element::eval(int node) const
{
  dolfin_assert(element);
  return element->eval(node);
}
//-----------------------------------------------------------------------------
void Element::update(real u0)
{

  dolfin_assert(element);

  element->update(u0);
}
//-----------------------------------------------------------------------------
void Element::update(RHS& f)
{
  dolfin_assert(element);

  element->update(f);
}
//-----------------------------------------------------------------------------
int Element::within(real t) const
{
  dolfin_assert(element);  
  return element->within(t);
}
//-----------------------------------------------------------------------------
bool Element::within(TimeSlab* timeslab) const
{
  dolfin_assert(element);
  return element->within(timeslab);
}
//-----------------------------------------------------------------------------
real Element::starttime() const
{
  //dolfin_segfault();
  dolfin_assert(element);
  return element->starttime();
}
//-----------------------------------------------------------------------------
real Element::endtime() const
{
  dolfin_assert(element);
  return element->endtime();
}
//-----------------------------------------------------------------------------
real Element::timestep() const
{
  dolfin_assert(element);
  return element->timestep();
}
//-----------------------------------------------------------------------------
real Element::newTimeStep() const
{
  dolfin_assert(element);
  return element->newTimeStep();
}
//-----------------------------------------------------------------------------
//void Element::operator=(Element& element)
//{
//  this->element = element.element;
//}
