// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Element.h>
#include <dolfin/Component.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Component::Component()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Component::Component(int size) : elements(size)
{
  //dolfin_segfault();
  //dolfin_assert(size > 0);
}
//-----------------------------------------------------------------------------
Component::~Component()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real Component::operator() (int node, real t, TimeSlab* timeslab)
{
  dolfin_debug("foo");
  dolfin::cout << "node: " << node << dolfin::endl;
  dolfin::cout << "t: " << t << dolfin::endl;

  // Step to correct position
  Element element = findpos(t);

  // Evaluation for element within the given time slab
  if ( element.within(timeslab) )
    return element.eval(node);

  // Evaluation for other elements
  return element.eval(t);
}
//-----------------------------------------------------------------------------
real Component::operator() (real t)
{
  Element element = elements.back();

  return element.eval(t);
}
//-----------------------------------------------------------------------------
int Component::add(Element& element, real t1)
{
  dolfin_debug1("this: %p", this);
  dolfin_debug1("t1: %lf", t1);

  // Estimate the number of elements
  //int n = next + ceil_int( (t1 - element.starttime()) / element.timestep() );

  dolfin_debug1("element.starttime: %lf", element.starttime());
  dolfin_debug1("element.timestep: %lf", element.timestep());
  dolfin_debug1("elements.size: %d", elements.size());
  
  elements.push_back(element);

  // Increase the size of the list if necessary
  //if ( n > elements.size() )
  //elements.resize(n);

  // Add the slab to the list
  //elements(next++) = element;

  return elements.size();
}
//-----------------------------------------------------------------------------
Element& Component::last()
{
  //dolfin_assert(next > 0);
  //dolfin_assert(next <= elements.size());

  return elements.back();
}
//-----------------------------------------------------------------------------
Element Component::findpos(real t)
{
  dolfin_debug("findpos");

  dolfin::cout << "Looking for t: " << t << dolfin::endl;

  dolfin_assert(elements.size() > 0);
  dolfin_assert(elements.front().starttime() <= t);
  dolfin_assert(elements.back().endtime() >= t);

  dolfin::cout << "Component range: " << elements.front().starttime() <<
    "-" << elements.back().endtime() << dolfin::endl;

  typedef std::vector<Element>::iterator ElementIterator;

  ElementIterator target;

  static Element dummy;
  LessElement comp(dummy, t);

  target = upper_bound(elements.begin(), elements.end(), dummy, comp);

  dolfin_debug("found");

  if(target == elements.end())
    --target;

  dolfin::cout << "Found element at: " << (*target).starttime() << "-" <<
    (*target).endtime() << dolfin::endl;


  return *target;

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
Component::LessElement::LessElement(Element &dummy, real t) : dummy(dummy),
							      t(t)
{
}
//-----------------------------------------------------------------------------
bool Component::LessElement::operator()(const Element &x, const Element &y)
{
  if(&x == &dummy)
    return t < y.starttime();
  else if(&y == &dummy)
    return x.starttime() < t;
  else
    return x.starttime() < y.starttime();
}
//-----------------------------------------------------------------------------
