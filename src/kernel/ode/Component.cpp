// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
Component::Component(int size) : elements(size), u0(0)
{
  //dolfin_segfault();
  //dolfin_assert(size > 0);
}
//-----------------------------------------------------------------------------
Component::~Component()
{
  // Delete elements
  for (int i = 0; i < elements.size(); i++)
    delete elements[i];
}
//-----------------------------------------------------------------------------
real Component::operator() (int node, real t, TimeSlab* timeslab)
{
  //dolfin_debug("foo");

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
real Component::operator()(real t)
{
  //dolfin_debug("foo");

  //dolfin::cout << "node: " << node << dolfin::endl;
  //dolfin::cout << "Looking for t: " << t << dolfin::endl;

  if(elements.size() > 0)
  {
    real t0 = elements.front()->starttime();
    real t1 = elements.back()->endtime();

    //dolfin::cout << "Component range: " << t0 << "-" << t1 << dolfin::endl;

    if(t == t0)
    {
      dolfin_debug("u0");
      return u0;
    }
    else if(t > t1)
    {
      // Should extrapolate
      // I don't think our sceheme requires it however

      dolfin_debug("after end");
      return u0;
    }
    else if(t < t0)
    {
      // Shouldn't ever get here

      dolfin_debug("before start");
      return u0;
    }
    else
    {
      Element *element = findpos(t);

      return element->eval(t);
    }
  }
  else
  {
    // Range empty

    dolfin_debug("Range empty");

    return u0;
  }
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
    Element &prevelement = last();
    real u0i = prevelement.eval();
    element->update(u0i);
  }
  
  // Add element to list
  elements.push_back(element);

  dolfin_debug1("this: %p", this);

  dolfin_debug1("element->starttime: %lf", element->starttime());
  dolfin_debug1("element->endtime: %lf", element->endtime());
  dolfin_debug1("element->timestep: %lf", element->timestep());
  dolfin_debug1("elements.size: %d", elements.size());

  return element;
}
//-----------------------------------------------------------------------------
Element& Component::last()
{
  //dolfin_assert(next > 0);
  //dolfin_assert(next <= elements.size());

  return *(elements.back());
}
//-----------------------------------------------------------------------------
Element *Component::findpos(real t)
{
  /// Find element through binary search

  dolfin_debug("findpos");


  //dolfin_assert(elements.size() > 0);
  //dolfin_assert(elements.front()->starttime() <= t);
  //dolfin_assert(elements.back()->endtime() >= t);

  typedef std::vector<Element *>::iterator ElementIterator;

  ElementIterator target;

  static Element *dummy = 0;
  LessElement comp(dummy, t);

  /// upper_bound produces an iterator target such that for all
  /// iterators i in [elements.begin(), target), t <= i->endtime() is
  /// false. In other words, t <= target->endtime() is true, while t <=
  /// (target - 1)->endtime() is false

  target = upper_bound(elements.begin(), elements.end(), dummy, comp);
  //target = lower_bound(elements.begin(), elements.end(), dummy, comp);

  //if(target != elements.begin())
  //{
  //  dolfin_debug("check");

  //  dolfin_debug1("t <= target->endtime(): %d", comp(dummy, *target));
  //  dolfin_debug1("t <= (target - 1)->endtime(): %d", comp(dummy,
  //							   *(target - 1)));
  //}

  if(target == elements.end())
  {
    dolfin_debug("found end");
    //target--;
    return 0;
  }
  else
  {
    dolfin_debug("found");
    dolfin::cout << "Found element at: " << (*target)->starttime() << "-" <<
      (*target)->endtime() << dolfin::endl;
    return *target;
  }


  /*
  Element element = elements(current);

  // Check if we are already at the correct position
  if ( element.within(t) )
    return element;

  // Try to compute the position (correct if the time steps for this
  // component are constant within the time slab)
  
  //int resolution = 1000;
  int n = elements.size(); // What if we don't use all elements?
  //current = max((t*n) / resolution, n);

  current = 0;

  int res = 0;

  element = elements(current);

  if ( (res = (element.within(t))) == 0 )
    return element;

  // If we couldn't find the position, do a linear search
  
  if ( res < 0 ) {

    // Step backwards
    for (; current > 0; current--)
      if ( (elements(current).within(t)) == 0 )
	return elements(current);

  }
  else {

    // Step forwards
    for (; current < (n-1); current--)
      if ( (elements(current).within(t)) == 0 )
	return elements(current);

  }

  return element;
  */
}
//-----------------------------------------------------------------------------
int Component::size()
{
  return elements.size();
}
//-----------------------------------------------------------------------------
Component::LessElement::LessElement(Element *dummy, real t) : dummy(dummy),
							      t(t)
{
}
//-----------------------------------------------------------------------------
bool Component::LessElement::operator()(const Element *x, const Element *y)
{
  if(x == dummy)
    return t <= y->endtime();
  else if(y == dummy)
    return x->endtime() <= t;
  else
    return x->endtime() <= y->endtime();
}
//-----------------------------------------------------------------------------
