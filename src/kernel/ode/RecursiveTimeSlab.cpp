// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Updates by Johan Jansson 2003

#include <iostream>
#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/Element.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/Partition.h>
#include <dolfin/RHS.h>
#include <dolfin/Solution.h>
#include <dolfin/RecursiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RecursiveTimeSlab::RecursiveTimeSlab(Element::Type type, unsigned int q,
				     real t0, real t1, Solution& u, RHS& f,
				     Adaptivity& adaptivity,
				     FixedPointIteration& fixpoint,
				     Partition& partition, int offset) :
  TimeSlab(t0, t1)
{
  // Create the time slab
  create(type, q, u, f, adaptivity, fixpoint, partition, offset);
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
real RecursiveTimeSlab::update(FixedPointIteration& fixpoint)
{
  // First update the time slabs
  real ds = updateTimeSlabs(fixpoint);

  // Then update the elements
  real de = updateElements(fixpoint);

  return std::max(ds, de);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::reset(Solution& u)
{
  // First reset the time slabs
  resetTimeSlabs(u);

  // Then reset the elements
  resetElements(u);
}
//-----------------------------------------------------------------------------
real RecursiveTimeSlab::computeMaxRd(Solution& u, RHS& f)
{
  // First check time slabs
  real rs = computeMaxRdTimeSlabs(u, f);

  // Then check the elements
  real re = computeMaxRdElements(u, f);

  return std::max(rs, re);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::create(Element::Type type, unsigned int q,
			       Solution& u, RHS& f,
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
    createTimeSlabs(type, q, u, f, adaptivity, fixpoint, partition, end);

  // Create elements for the components with large time steps
  createElements(type, q, u, f, adaptivity, fixpoint, partition, offset, end);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::createTimeSlabs(Element::Type type, unsigned int q,
					Solution& u, RHS& f, 
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
      new RecursiveTimeSlab(type, q, t, t1, u, f, adaptivity, fixpoint, 
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
void RecursiveTimeSlab::createElements(Element::Type type, unsigned int q,
				       Solution& u, RHS& f,
				       Adaptivity& adaptivity,
				       FixedPointIteration& fixpoint,
				       Partition& partition,
				       int offset, int end)

{
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
  updateElements(fixpoint);

  // Compute residuals and new time steps
  computeResiduals(f, adaptivity);
}
//-----------------------------------------------------------------------------
real RecursiveTimeSlab::updateTimeSlabs(FixedPointIteration& fixpoint)
{
  real dmax = 0.0;

  // Update time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    dmax = std::max(dmax, timeslabs[i]->update(fixpoint));
  
  return dmax;
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::resetTimeSlabs(Solution& u)
{
  // Reset time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    timeslabs[i]->reset(u);
}
//-----------------------------------------------------------------------------
real RecursiveTimeSlab::computeMaxRdTimeSlabs(Solution& u, RHS& f)
{
  real rdmax = 0;

  for (unsigned int i = 0; i < timeslabs.size(); i++)
    rdmax = std::max(rdmax, fabs(timeslabs[i]->computeMaxRd(u, f)));

  return rdmax;
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::computeResiduals(RHS& f, Adaptivity& adaptivity)
{
  // Get tolerance and maximum time step
  real TOL = adaptivity.tolerance();
  real kmax = adaptivity.maxstep();
  bool kfixed = adaptivity.fixed();

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
    adaptivity.regulator(element->index()).update(k, kmax, kfixed);
   }
 }
//-----------------------------------------------------------------------------
