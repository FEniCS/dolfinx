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
void RecursiveTimeSlab::update(FixedPointIteration& fixpoint)
{
  // First update the time slabs
  updateTimeSlabs(fixpoint);

  // Then iterate on the elements
  fixpoint.iterate(group);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::reset(FixedPointIteration& fixpoint)
{
  // First reset the time slabs
  resetTimeSlabs(fixpoint);

  // Then reset the elements
  fixpoint.reset(group);
}
//-----------------------------------------------------------------------------
bool RecursiveTimeSlab::leaf() const
{
  return timeslabs.size() == 0;
}
//-----------------------------------------------------------------------------
real RecursiveTimeSlab::elementResidualL2(FixedPointIteration& fixpoint)
{
  // Compute L2 norm for time slabs
  real r = 0.0;
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    r += sqr(timeslabs[i]->elementResidualL2(fixpoint));

  // Compute L2 norm for elements
  r += sqr(fixpoint.residual(group));

  return sqrt(r);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::show(unsigned int depth) const
{
  for (unsigned int i = 0; i < depth; i++)
    cout << "  ";
  
  cout << "Time slab at [" << starttime() << " " << endtime() << "]: "
       << group.size() << " element(s) and "
       << timeslabs.size() << " time slab(s)" << endl;
  
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
  // Create elements
  for (int i = offset; i < end; i++)
  {
    // Create element
    unsigned int index = partition.index(i);
    Element* element = u.createElement(u.method(index), u.order(index), 
				       index, t0, t1);
    
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
void RecursiveTimeSlab::updateTimeSlabs(FixedPointIteration& fixpoint)
{
  // Update time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    timeslabs[i]->update(fixpoint);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::resetTimeSlabs(FixedPointIteration& fixpoint)
{
  // Reset time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    timeslabs[i]->reset(fixpoint);
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
