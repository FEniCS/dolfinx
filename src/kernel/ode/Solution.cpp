// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Solution.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Solution::Solution(const ODE& ode, ElementData& elmdata) : 
  elmdata(elmdata), u0(ode.size())
{
  // Set initial data
  for (unsigned int i = 0; i < u0.size(); i++)
    u0[i] = ode.u0(i);
}
//-----------------------------------------------------------------------------
Solution::~Solution()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real Solution::operator() (unsigned int i, real t)
{
  dolfin_assert(i < u0.size());

  // Note: the logic of this function is nontrivial...

  // First check if the initial value is requested. We don't want to ask
  // elmdata for the element, since elmdata might go looking for the element
  // on disk if it is not available.

  if ( t == t0 )
    return u0[i];

  // Then try to find the element and return the value if we found it
  Element* element = elmdata.element(i,t);
  if ( element )
    return element->value(t);

  // If we couldn't find the element return initial value (extrapolation)
  return u0[i];
}
//-----------------------------------------------------------------------------
