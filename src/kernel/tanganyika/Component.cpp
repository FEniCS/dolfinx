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
Component::Component(int size)
{
  dolfin_segfault();
  dolfin_assert(size > 0);

  // Initialize the list of elements to given size (which is only a guess)
  elements.init(size);
}
//-----------------------------------------------------------------------------
Component::~Component()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Component::init(int size)
{
  elements.init(size);
}
//-----------------------------------------------------------------------------
real Component::operator() (int node, real t, TimeSlab* timeslab)
{
  // Step to correct position
  Element element = findpos(t);

  // Evaluation for element within the given time slab
  if ( element.within(timeslab) )
    return element.eval(node);

  // Evaluation for other elements
  return element.eval(t);
}
//-----------------------------------------------------------------------------
int Component::add(Element& element, real t1)
{
  // Estimate the number of elements
  int n = next + ceil_int( (t1 - element.starttime()) / element.timestep() );
  
  // Increase the size of the list if necessary
  if ( n > elements.size() )
    elements.resize(n);

  // Add the slab to the list
  elements(next++) = element;
}
//-----------------------------------------------------------------------------
Element& Component::last()
{
  dolfin_assert(next > 0);
  dolfin_assert(next <= elements.size());

  return elements(next-1);
}
//-----------------------------------------------------------------------------
Element Component::findpos(real t)
{
  Element element = elements(current);

  // Check if we are already at the correct position
  if ( element.within(t) )
    return element;

  // Try to compute the position (correct if the time steps for this
  // component are constant within the time slab)
  
  int resolution = 1000;
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

}
//-----------------------------------------------------------------------------
