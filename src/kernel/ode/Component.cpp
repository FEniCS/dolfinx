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
Component::Component()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Component::~Component()
{
  clear();
}
//-----------------------------------------------------------------------------
Element* Component::createElement(const Element::Type type, real t0, real t1,
				  int q, int index)
{
  Element* element = 0;

  // Create element
  switch (type) {
  case Element::cg:
    element = new cGqElement(t0, t1, q, index);
    break;
  case Element::dg:
    element = new dGqElement(t0, t1, q, index);
    break;
  default:
    dolfin_error1("Unknown element type: %d.", type);
  }
  
  // Add element to list
  elements.push_back(element);

  return element;
}
//-----------------------------------------------------------------------------
Element* Component::element(real t)
{
  return findpos(t);
}
//-----------------------------------------------------------------------------
Element* Component::last()
{
  if ( elements.empty() )
    return 0;

  return elements.back();
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
}
//-----------------------------------------------------------------------------
unsigned int Component::size() const
{
  return elements.size();
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
