// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2003.

#include <iostream>
#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Element.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/Partition.h>
#include <dolfin/RHS.h>
#include <dolfin/Solution.h>
#include <dolfin/FixedPointIteration.h>
#include <dolfin/RecursiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RecursiveTimeSlab::RecursiveTimeSlab(real t0, real t1, Solution& u, RHS& f,
				     Adaptivity& adaptivity,
				     FixedPointIteration& fixpoint,
				     Partition& partition, int offset) :
  TimeSlab(t0, t1)
{
  // Create the time slab
  create(u, f, adaptivity, fixpoint, partition, offset);
}
//-----------------------------------------------------------------------------
RecursiveTimeSlab::~RecursiveTimeSlab()
{
  // Delete the time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
  {
    if ( timeslabs[i] )
      delete timeslabs[i];
    timeslabs[i] = 0;
  }
}
//-----------------------------------------------------------------------------
bool RecursiveTimeSlab::accept(RHS& f, real TOL)
{
  // Check time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    if ( !timeslabs[i]->accept(f, TOL) )
      return false;

  // Check elements
  for (ElementIterator element(group); !element.end(); ++element)
  {
    // Compute residual
    real r = element->computeResidual(f);

    // Check element
    if ( !element->accept(TOL, r) )
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::countElementGroups(unsigned int& size)
{
  // Count number of element groups for time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    timeslabs[i]->countElementGroups(size);
  
  // Count element groups contained in this time slab
  ++size;
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::addElementGroups(NewArray<ElementGroup*>& groups,
					 unsigned int& pos)
{
  // Add element groups contained in time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    timeslabs[i]->addElementGroups(groups, pos);

  // Add the element group contained in this time slab
  groups[pos++] = &group;
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::show(unsigned int depth) const
{
  for (unsigned int i = 0; i < depth; i++)
    cout << "  ";
  
  dolfin_info("Time slab at [%f %f].",  starttime(), endtime());
  
  for (unsigned int i = 0; i < timeslabs.size(); i++)
  {
    TimeSlab* timeslab = timeslabs[i];
    dolfin_assert(timeslab);
    timeslab->show(depth + 1);
  }
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::create(Solution& u, RHS& f,
			       Adaptivity& adaptivity,
			       FixedPointIteration& fixpoint,
			       Partition& partition, int offset)
{
  int end = 0;
  real K = 0.0;
  
  // Update partitition 
  partition.update(offset, end, K, adaptivity);
  
  // Adjust and set the size of this time slab 
  setsize(K, adaptivity);

  // Create time slabs for the components with small time steps
  if (end < partition.size())
    createTimeSlabs(u, f, adaptivity, fixpoint, partition, end);

  // Create elements for the components with large time steps
  createElements(u, f, adaptivity, fixpoint, partition, offset, end);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::createTimeSlabs(Solution& u, RHS& f, 
					Adaptivity& adaptivity,
					FixedPointIteration& fixpoint,
					Partition& partition, int offset)
{
  // Current position
  real t = t0;

  // Create the list of time slabs
  while ( true )
  {
    // Create a new time slab
    TimeSlab* timeslab = 
      new RecursiveTimeSlab(t, t1, u, f, adaptivity, fixpoint, 
			    partition, offset);
    
    // Add the new time slab to the list
    timeslabs.push_back(timeslab);
    
    // Check if we are done
    if(timeslab->finished())
      break;
    
    // Step to next time slab
    t = timeslab->endtime();
  }
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::createElements(Solution& u, RHS& f,
				       Adaptivity& adaptivity,
				       FixedPointIteration& fixpoint,
				       Partition& partition,
				       int offset, int end)

{
  // Get length of this time slab
  real k = length();

  // Create elements
  for (int i = offset; i < end; i++)
  {
    // Create element
    unsigned int index = partition.index(i);
    Element* element = u.createElement(u.method(index), u.order(index), 
				       index, t0, t1);
    
    // Update regulator for component
    if ( !adaptivity.fixed() )
      adaptivity.regulator(index).init(k);

    // Write debug info
    u.debug(*element, Solution::create);
    
    // Add element to array
    group.add(*element);
  }

  // Reset elements
  fixpoint.reset(group);

  // Iterate on the elements
  fixpoint.iterate(group);

  // Compute residuals and new time steps
  computeResiduals(f, adaptivity);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::computeResiduals(RHS& f, Adaptivity& adaptivity)
{
  // Get tolerance and maximum time step
  real TOL = adaptivity.tolerance();
  real kmax = adaptivity.maxstep();
  bool kfixed = adaptivity.fixed();

  // Compute residuals and new time steps
  for (ElementIterator element(group); !element.end(); ++element)
  {
    // Compute residual
    real r = element->computeResidual(f);

    // Compute new time step
    real k = element->computeTimeStep(TOL, r, kmax);

    // Update regulator
    adaptivity.regulator(element->index()).update(k, kmax, kfixed);
   }
 }
//-----------------------------------------------------------------------------
