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
SimpleTimeSlab::SimpleTimeSlab(Element::Type type, unsigned int q,
			       real t0, real t1, Solution& u, 
			       Adaptivity& adaptivity) : TimeSlab(t0, t1)
{
  create(type, q, u, adaptivity);
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
void SimpleTimeSlab::reset(Solution& u)
{
  resetElements(u);
}
//-----------------------------------------------------------------------------
real SimpleTimeSlab::computeMaxRd(Solution& u, RHS& f)
{
  return computeMaxRdElements(u, f);
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::create(Element::Type type, unsigned int q,
			    Solution& u, Adaptivity& adaptivity)
{
  // Get initial time step (same for all components)
  real k = adaptivity.regulator(0).timestep();

  // Set size of this time slab
  setsize(k, adaptivity);

  // Create elements
  for (unsigned int i = 0; i < u.size(); i++)
  {
    // Create element
    Element *element = u.createElement(type, q, i, t0, t1);
    
    // Write debug info
    u.debug(*element, Solution::create);

    // Add element to array
    elements.push_back(element);
  }
}
//-----------------------------------------------------------------------------
