// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>

#include <dolfin/dolfin_log.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/RHS.h>
#include <dolfin/Solution.h>
#include <dolfin/SimpleTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SimpleTimeSlab::SimpleTimeSlab(real t0, real t1, Solution& u, 
			       Adaptivity& adaptivity) : TimeSlab(t0, t1)
{
  create(u, adaptivity);
}
//-----------------------------------------------------------------------------
SimpleTimeSlab::~SimpleTimeSlab()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real SimpleTimeSlab::update(FixedPointIteration& fixpoint)
{
  return updateElements(fixpoint);
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::reset(FixedPointIteration& fixpoint)
{
  resetElements(fixpoint);
}
//-----------------------------------------------------------------------------
real SimpleTimeSlab::computeMaxRd(Solution& u, RHS& f)
{
  return computeMaxRdElements(u, f);
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::create(Solution& u, Adaptivity& adaptivity)
{
  // Use the minimal time step for all components
  real kmin = adaptivity.minstep();

  // Set size of this time slab
  setsize(kmin, adaptivity);

  // Create elements
  for (unsigned int i = 0; i < u.size(); i++)
  {
    // Create element
    Element *element = u.createElement(u.method(i), u.order(i), i, t0, t1);
    
    // Write debug info
    u.debug(*element, Solution::create);

    // Add element to array
    elements.push_back(element);
  }
}
//-----------------------------------------------------------------------------
