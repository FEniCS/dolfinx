// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Updates by Johan Jansson 2003

#include <iostream>

#include <dolfin/dolfin_log.h>
#include <dolfin/Element.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/Partition.h>
#include <dolfin/RHS.h>
#include <dolfin/Solution.h>
#include <dolfin/RecursiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RecursiveTimeSlab::RecursiveTimeSlab(real t0, real t1, Solution& u, RHS& f,
				     Adaptivity& adaptivity, 
				     Partition& partition, int offset) :
  TimeSlab(t0, t1)
{
  // Create the time slab
  create(u, f, adaptivity, partition, offset);
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
void RecursiveTimeSlab::update(Solution& u, RHS& f)
{
  // First update the time slabs
  updateTimeSlabs(u, f);

  // Then update the elements
  updateElements(u, f);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::create(Solution& u, RHS& f, Adaptivity& adaptivity,
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
    createTimeSlabs(u, f, adaptivity, partition, end);

  // Create elements for the components with large time steps
  createElements(u, f, adaptivity, partition, offset, end);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::createTimeSlabs(Solution& u, RHS& f, 
					Adaptivity& adaptivity,
					Partition& partition, int offset)
{
  // Current position
  real t = t0;

  // Create the list of time slabs
  while ( true )
  {
    // Create a new time slab
    TimeSlab* timeslab = 
      new RecursiveTimeSlab(t, t1, u, f, adaptivity, partition, offset);
    
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
				       Partition& partition,
				       int offset, int end)

{
  // FIXME: choose element and order here
  Element::Type type = Element::cg;
  int q = 1;

  // Create elements
  for (int i = offset; i < end; i++) {

    // Create element
    Element* element = u.createElement(type, q, partition.index(i), t0, t1);
    
    // Write debug info
    u.debug(*element, Solution::create);
    
    // Add element to array
    elements.push_back(element);
  }

  // Update elements
  updateElements(u, f);

  // Compute residuals and new time steps
  computeResiduals(f, adaptivity);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::updateTimeSlabs(Solution& u, RHS& f)
{
  // Update time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    timeslabs[i]->update(u, f);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::computeResiduals(RHS& f, Adaptivity& adaptivity)
{
  // Get tolerance and maximum time step
  real TOL = adaptivity.tolerance();
  real kmax = adaptivity.maxstep();

  // Compute residuals and new time steps
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get element
    Element* element = elements[i];

    // Compute residual
    real r = element->computeResidual(f);

    // Compute new time step
    real k = element->computeTimeStep(TOL, r, kmax);

    // Update regulator
    adaptivity.regulator(element->index()).update(k, kmax);
   }
 }
//-----------------------------------------------------------------------------
