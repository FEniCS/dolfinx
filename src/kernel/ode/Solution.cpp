// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/RHS.h>
#include <dolfin/Function.h>
#include <dolfin/ElementData.h>
#include <dolfin/Solution.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Solution::Solution(ODE& ode, Function& u) :
  elmdata(u.elmdata()), u0(ode.size()), t0(0)
{
  // Get parameters
  _debug  = dolfin_get("debug time steps");

  // Set initial data
  for (unsigned int i = 0; i < u0.size(); i++)
    u0[i] = ode.u0(i);
  
  // Open debug file
  if ( _debug )
    file.open("timesteps.debug", std::ios::out);

  // Set name and label of solution (same as function)
  rename(u.name(), u.label());
}
//-----------------------------------------------------------------------------
Solution::~Solution()
{
  // Close debug file
  if ( _debug )
    file.close();
}
//-----------------------------------------------------------------------------
Element* Solution::createElement(Element::Type type, 
				 unsigned int q, unsigned int index, 
				 real t0, real t1)
{
  return elmdata.createElement(type, q, index, t0, t1);
}
//-----------------------------------------------------------------------------
Element* Solution::element(unsigned int i, real t)
{
  return elmdata.element(i,t);
}
//-----------------------------------------------------------------------------
Element* Solution::first(unsigned int i)
{
  return elmdata.first(i);
}
//-----------------------------------------------------------------------------
Element* Solution::last(unsigned int i)
{
  return elmdata.last(i);
}
//-----------------------------------------------------------------------------
real Solution::operator() (unsigned int i, real t)
{
  return u(i,t);
}
//-----------------------------------------------------------------------------
real Solution::u(unsigned int i, real t)
{
  dolfin_assert(i < u0.size());
  
  // Note: the logic of this function is nontrivial.
  
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
real Solution::k(unsigned int i, real t)
{
  Element* element = elmdata.element(i,t);
  if ( element )
    return element->timestep();
  
  return 0.0;
}
//-----------------------------------------------------------------------------
real Solution::r(unsigned int i, real t, RHS& f)
{
  Element* element = elmdata.element(i,t);
  if ( element )
    return element->computeResidual(f);

  return 0.0;
}
//-----------------------------------------------------------------------------
unsigned int Solution::size() const
{
  return elmdata.size();
}
//-----------------------------------------------------------------------------
void Solution::shift(real t0)
{
  for (unsigned int i = 0; i < elmdata.size(); i++)
  {
    // Get last element
    Element* element = elmdata.last(i);
    dolfin_assert(element);
    dolfin_assert(element->endtime() == t0);

    // Update initial value
    u0[i] = element->endval();
  }
  
  // Update time for initial values
  this->t0 = t0;

  // Tell element data to create a new block next time
  elmdata.shift();
}
//-----------------------------------------------------------------------------
void Solution::reset()
{
  elmdata.dropLast();
}
//-----------------------------------------------------------------------------
void Solution::debug(Element& element, Action action)
{
  if ( !_debug )
    return;

  // Write debug info to file
  file << action << " "
       << element.index() << " " 
       << element.starttime() << " " 
       << element.endtime() << "\n";
}
//-----------------------------------------------------------------------------
