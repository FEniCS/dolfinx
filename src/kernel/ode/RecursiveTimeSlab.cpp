// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Updates by Johan Jansson 2003

#include <iostream>

#include <dolfin/dolfin_log.h>
#include <dolfin/Element.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/Partition.h>
#include <dolfin/RHS.h>
#include <dolfin/RecursiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RecursiveTimeSlab::RecursiveTimeSlab(real t0, real t1, RHS& f,
				     TimeSlabData& data, Partition& partition, 
				     int offset) : TimeSlab(t0, t1)
{
  // Create the time slab
  create(f, data, partition, offset);
}
//-----------------------------------------------------------------------------
RecursiveTimeSlab::~RecursiveTimeSlab()
{
  // Delete the time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
  {
    delete timeslabs[i];
    timeslabs[i] = 0;
  }
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::update(RHS& f, TimeSlabData& data)
{
  // First update the time slabs
  updateTimeSlabs(f, data);

  // Then update the elements
  updateElements(f, data);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::create(RHS& f, TimeSlabData& data,
			       Partition& partition, int offset)
{
  dolfin_start();
  dolfin_debug1("Creating time slab with offset = %d", offset);

  int end = 0;
  real K = 0.0;

  // Update partitition 
  partition.update(offset, end, K, data);

  // Adjust and set the size of this time slab 
  setsize(K, data);

  // Create time slabs for the components with small time steps
  if (end < partition.size())
    createTimeSlabs(f, data, partition, end);

  // Create elements for the components with large time steps
  createElements(f, data, partition, offset, end);
  
  dolfin_end();
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::createTimeSlabs(RHS& f, TimeSlabData& data, 
					Partition& partition, int offset)
{
  // Current position
  real t = t0;

  // Create the list of time slabs
  while ( true )
  {
    // Create a new time slab
    TimeSlab* timeslab = new RecursiveTimeSlab(t, t1, f, data, partition, offset);
    
    // Add the new time slab to the list
    timeslabs.push_back(timeslab);

    // Check if we are done
    if(timeslab->finished())
      break;
    
    // Step to next time slab
    t = timeslab->endtime();
  }

  // Remove unused time slabs
  //timeslabs.resize();
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::createElements(RHS& f, TimeSlabData& data,
				       Partition& partition, int offset, int end)
{
  // Instead of storing a list of elements, we store two iterators to
  // the list of elements in TimeSlabData.
  
  // FIXME: choose element and order here
  Element::Type type = Element::cg;
  //Element::Type type = Element::dg;

  int q = 1;
  //int q = 0;

  // Create elements
  for (int i = offset; i < end; i++) {

    // Create element
    Element *element = data.createElement(type, q, partition.index(i), this);

    // Write debug info
    data.debug(*element, TimeSlabData::create);
    
    // Add element to array
    elements.push_back(element);
  }

  // Update elements
  updateElements(f, data);

  // Compute residuals and new time steps
  computeResiduals(f, data);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::updateTimeSlabs(RHS& f, TimeSlabData& data)
{
  // Update time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    timeslabs[i]->update(f, data);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::updateElements(RHS& f, TimeSlabData& data)
{
  // Update initial condition for elements
    updateu0(data);

  // Update elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
        
    // Update element
    element->update(f);
    
    // Write debug info
    data.debug(*element, TimeSlabData::update);
  }
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::updateu0(TimeSlabData& data)
{
  // FIXME: Can we be sure that we get the correct value? Maybe we obtain
  // FIXME: the value from this element and not the previous? We should
  // FIXME: probably ask explicitly for the last value of the previous
  // FIXME: element.

  // Update initial values
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    Element* e = elements[i];
    real u0 = data.component(e->index())(e->starttime());
    e->update(u0);
  }  
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::computeResiduals(RHS& f, TimeSlabData& data)
{
  // Get tolerance and maximum time step
  real TOL = data.tolerance();
  real kmax = data.maxstep();

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
    data.regulator(element->index()).update(k, data.maxstep());
   }
 }
//-----------------------------------------------------------------------------
