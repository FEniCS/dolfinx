// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Updates by Johan Jansson 2003

#include <iostream>

#include <dolfin/dolfin_log.h>
#include <dolfin/Element.h>
#include <dolfin/TimeSteppingData.h>
#include <dolfin/Partition.h>
#include <dolfin/RHS.h>
#include <dolfin/RecursiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RecursiveTimeSlab::RecursiveTimeSlab(real t0, real t1, RHS& f,
				     TimeSteppingData& data,
				     Partition& partition,
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
void RecursiveTimeSlab::update(RHS& f, TimeSteppingData& data)
{
  // First update the time slabs
  updateTimeSlabs(f, data);

  // Then update the elements
  updateElements(f, data);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::create(RHS& f, TimeSteppingData& data,
			       Partition& partition, int offset)
{
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
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::createTimeSlabs(RHS& f, TimeSteppingData& data,
					Partition& partition, int offset)
{
  // Current position
  real t = t0;

  // Create the list of time slabs
  while ( true )
  {
    // Create a new time slab
    TimeSlab* timeslab = new RecursiveTimeSlab(t, t1, f, data, 
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
void RecursiveTimeSlab::createElements(RHS& f, TimeSteppingData& data,
				       Partition& partition, int offset, int end)
{
  // FIXME: choose element and order here
  Element::Type type = Element::cg;
  int q = 1;

  // Create elements
  for (int i = offset; i < end; i++) {

    // Create element
    Element *element = data.createElement(type, t0, t1, q, partition.index(i));

    // Write debug info
    data.debug(*element, TimeSteppingData::create);
    
    // Add element to array
    elements.push_back(element);
  }

  // Update elements
  updateElements(f, data);

  // Compute residuals and new time steps
  computeResiduals(f, data);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::updateTimeSlabs(RHS& f, TimeSteppingData& data)
{
  // Update time slabs
  for (unsigned int i = 0; i < timeslabs.size(); i++)
    timeslabs[i]->update(f, data);
}
//-----------------------------------------------------------------------------
void RecursiveTimeSlab::computeResiduals(RHS& f, TimeSteppingData& data)
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
    data.regulator(element->index()).update(k, kmax);
   }
 }
//-----------------------------------------------------------------------------
