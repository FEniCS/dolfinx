// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson, 2003.

#include <iostream>

#include <dolfin/dolfin_log.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/RHS.h>
#include <dolfin/Solution.h>
#include <dolfin/FixedPointIteration.h>
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
void SimpleTimeSlab::update(FixedPointIteration& fixpoint)
{
  fixpoint.iterate(group);
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::reset(FixedPointIteration& fixpoint)
{
  fixpoint.reset(group);
}
//-----------------------------------------------------------------------------
bool SimpleTimeSlab::leaf() const
{
  return true;
}
//-----------------------------------------------------------------------------
real SimpleTimeSlab::elementResidualL2(FixedPointIteration& fixpoint)
{
  return fixpoint.residual(group);
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::countElementGroups(unsigned int& size)
{
  // A simple time slab contains only one element group
  size = 1;
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::addElementGroups(NewArray<ElementGroup*>& groups,
				      unsigned int& pos)
{
  // Add the element group
  groups[pos++] = &group;
}
//-----------------------------------------------------------------------------
void SimpleTimeSlab::show(unsigned int depth) const
{
  for (unsigned int i = 0; i < depth; i++)
    cout << "  ";

  cout << "Time slab at [" << starttime() << " " << endtime() << "]: "
       << group.size() << " element(s)" << endl;
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
    group.add(*element);
  }
}
//-----------------------------------------------------------------------------
