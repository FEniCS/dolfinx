// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <functional>
#include <algorithm>

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/cGqElement.h>
#include <dolfin/dGqElement.h>
#include <dolfin/Component.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Component::Component() : u0(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Component::Component(int size) : elements(size), u0(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Component::~Component()
{
  clear();
}
//-----------------------------------------------------------------------------
Element& Component::element(real t)
{
  Element* element = findpos(t);
  dolfin_assert(element);

  return *element;
}
//-----------------------------------------------------------------------------
Element& Component::last()
{
  dolfin_assert(elements.size() > 0);

  return *(elements.back());
}
//-----------------------------------------------------------------------------
int Component::size() const
{
  return elements.size();
}
//-----------------------------------------------------------------------------
real Component::value(real t)
{
  Element* element = findpos(t);
  dolfin_assert(element);

  return element->value(t);
}
//-----------------------------------------------------------------------------
real Component::timestep(real t)
{
  Element* element = findpos(t);
  dolfin_assert(element);

  return element->timestep();
}
//-----------------------------------------------------------------------------
real Component::residual(real t, RHS& f)
{
  Element* element = findpos(t);
  dolfin_assert(element);

  return element->computeResidual(f);
}
//-----------------------------------------------------------------------------
real Component::operator()(real t)
{
  //dolfin_debug("foo");

  //dolfin::cout << "node: " << node << dolfin::endl;
  //dolfin::cout << "Looking for t: " << t << dolfin::endl;

  if (elements.size() > 0)
  {
    real t0 = elements.front()->starttime();
    real t1 = elements.back()->endtime();

    //dolfin::cout << "Component range: " << t0 << "-" << t1 << dolfin::endl;

    if (t == t0)
    {
      //dolfin_debug("u0");
      return u0;
    }
    else if (t > t1)
    {
      // Should extrapolate
      // I don't think our sceheme requires it however

      //dolfin_debug("after end");
      return u0;
    }
    else if (t < t0)
    {
      // Shouldn't ever get here
      dolfin_error("Request for element value at t < t0");
      //dolfin_debug("before start");
      return u0;
    }
    else
    {
      const Element* element = findpos(t);
      dolfin_assert(element);
      return element->value(t);
    }
  }
  else
  {
    // Range empty

    //dolfin_debug("Range empty");

    return u0;
  }
}
//-----------------------------------------------------------------------------
real Component::operator() (int node, real t, TimeSlab* timeslab)
{
  // FIXME: Make this one work

  //dolfin::cout << "node: " << node << dolfin::endl;
  //dolfin::cout << "t: " << t << dolfin::endl;

  // Step to correct position
  //Element& element = findpos(t);

  // Evaluation for element within the given time slab
  //if ( element.within(timeslab) )
  //return element.eval(node);

  // Evaluation for other elements
  //return element.eval(t);
  return operator()(t);
}
//-----------------------------------------------------------------------------
Element* Component::createElement(const Element::Type type, int q, int index, 
				  TimeSlab* timeslab)
{
  Element* element = 0;

  // Create element
  switch (type) {
  case Element::cg:
    element = new cGqElement(q, index, timeslab);
    break;
  case Element::dg:
    element = new dGqElement(q, index, timeslab);
    break;
  default:
    dolfin_error1("Unknown element type: %d.", type);
  }
  
  // Set initial value
  if(size() == 0)
  {
    // First element, get u0 from explicit value
    element->update(u0);
  }
  else
  {
    // Not the First element, get u0 from previous element
    element->update(last().endval());
  }
  
  // Add element to list
  elements.push_back(element);

  return element;
}
//-----------------------------------------------------------------------------
void Component::clear()
{
  // Delete elements
  for (unsigned int i = 0; i < elements.size(); i++) {
    delete elements[i];
    elements[i] = 0;
  }
  
  elements.clear();
  u0 = 0.0;
}
//-----------------------------------------------------------------------------
Element* Component::findpos(real t) 
{
  /// Find element through binary search

  typedef std::vector<Element *>::iterator ElementIterator;

  ElementIterator target;

  static Element* dummy = 0;
  Less comp(dummy, t);

  /// The function upper_bound() produces an iterator target such that
  /// i->endtime() is false for all iterators in the interval
  /// [elements.begin(), target). In other words, t <= target->endtime()
  /// is true, while t <= (target - 1)->endtime() is false.

  target = upper_bound(elements.begin(), elements.end(), dummy, comp);

  if (target == elements.end())
    return 0;

  return *target;
}
//-----------------------------------------------------------------------------
Component::Less::Less(Element *dummy, real t) : 
  dummy(dummy), t(t)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool Component::Less::operator()(const Element* x, const Element* y)
{
  if (x == dummy)
    return t <= y->endtime();
  else if(y == dummy)
    return x->endtime() <= t;
  else
    return x->endtime() <= y->endtime();
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, 
				      const Component& component)
{
  stream << "[ Component of size " << component.size() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
