// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Element.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/Partition.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/RHS.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlab::TimeSlab(real t0, real t1, RHS& f, 
		   TimeSlabData& data, Partition& partition, int offset)
{
  dolfin_debug2("Creating time slab: [%f %f]", t0, t1);

  // Set data for this time slab
  this->t0 = t0;
  this->t1 = t1;
  reached_endtime = false;

  // If this is the top level time slab, notify TimeSlabData
  if ( offset == 0 )
  {
    data.setslab(this);
    dolfin_debug("setslab");
  }

  create(f, data, partition, offset);
}
//-----------------------------------------------------------------------------
TimeSlab::~TimeSlab()
{
  dolfin_debug("Clearing time slab");

  // Delete the time slabs
  for (int i = 0; i < timeslabs.size(); i++) {
    delete timeslabs[i];
  }
}
//-----------------------------------------------------------------------------
void TimeSlab::update(RHS& f)
{
  dolfin_info("Updating time slab");

  // First update the time slabs
  updateTimeSlabs(f);

  // Then update the elements
  updateElements(f);
}
//-----------------------------------------------------------------------------
bool TimeSlab::within(real t) const
{
  // Check if t is in the interval (t0,t1] = [t0 + eps, t1 + eps].
  // We need to make sure that we include the end-point. Otherwise
  // a round-off error may cause the next interval to be chosen,
  // which is not what we want, at least not for dG.

  t += DOLFIN_EPS;

  return (t0 <= t) && (t <= t1);
}
//-----------------------------------------------------------------------------
bool TimeSlab::finished() const
{
  return reached_endtime;
}
//-----------------------------------------------------------------------------
real TimeSlab::starttime() const
{
  return t0;
}
//-----------------------------------------------------------------------------
real TimeSlab::endtime() const
{
  return t1;
}
//-----------------------------------------------------------------------------
real TimeSlab::length() const
{
  return t1 - t0;
}
//-----------------------------------------------------------------------------
void TimeSlab::create(RHS& f, TimeSlabData& data,
		      Partition& partition, int offset)
{
  dolfin_debug1("offset: %d", offset);

  int end = 0;
  real K = 0.1;

  dolfin_debug("Computing partition");

  // Update the partition when necessary (not needed for the first time slab)
  partition.update(data, end);

  // Compute a partition into small and large time steps
  partition.partition(offset, end, K);

  dolfin_debug1("offset: %d", offset);
  dolfin_debug1("end: %d", end);

  dolfin_debug1("Adjusting time step K = %f", K);

  // Adjust and set the size of this time slab 
  setsize(K);

  dolfin_debug1("New end time = %f", t1);

  if(end < (f.size() - 1))
  {
    dolfin_debug("Create subslabs");

    // Create time slabs for the components with small time steps
    createTimeSlabs(f, data, partition, end);
  }

  dolfin_debug1("offset: %d", offset);

  // Create elements for the components with large time steps
  createElements(f, data, partition, offset);
}
//-----------------------------------------------------------------------------
void TimeSlab::createTimeSlabs(RHS& f, TimeSlabData& data, 
			       Partition& partition, int offset)
{
  dolfin_info("Creating time slabs");

  // Current position
  real t = t0;

  // Create the list of time slabs
  while ( true ) {


    // Create a new time slab
    TimeSlab* timeslab = new TimeSlab(t, t1, f, data, partition, offset);
    
    // Add the new time slab to the list
    add(timeslab);

    // Check if we are done
    if(timeslab->finished())
      break;
    
    // Step to next time slab
    t = timeslab->t1;
  }

  // Remove unused time slabs
  //timeslabs.resize();
}
//-----------------------------------------------------------------------------
void TimeSlab::createElements(RHS& f, TimeSlabData& data,
			      Partition& partition, int offset)
{
  // Instead of storing a list of elements, we store two iterators to
  // the list of elements in TimeSlabData.

  dolfin_info("Creating elements");

  // FIXME: choose element and order here
  Element::Type type = Element::cg;
  int q = 1;

  dolfin_debug1("slab: %p", this);
  

  // Create elements
  for (int i = offset; i < f.size(); i++)
  {
    Element e = data.createElement(type, q, partition.index(i), this);

    elements.push_back(e);
  }

  dolfin_debug("Update elements:");

  // Update elements
  updateElements(f);
}
//-----------------------------------------------------------------------------
void TimeSlab::updateTimeSlabs(RHS& f)
{
  for (int i = 0; i < timeslabs.size(); i++)
  {
    timeslabs[i]->update(f);
  }
}
//-----------------------------------------------------------------------------
void TimeSlab::updateElements(RHS& f)
{
  dolfin_debug1("elements: %d", elements.size());

  for(std::vector<Element>::iterator e = elements.begin();
      e != elements.end(); e++)
  {
    e->update(f);
    real value;
    value = e->eval(e->starttime());
    dolfin::cout << "element value at starttime: " << value << dolfin::endl;
    value = e->eval(e->endtime());
    dolfin::cout << "element value at endtime: " << value << dolfin::endl;

  }
}
//-----------------------------------------------------------------------------
void TimeSlab::setsize(real K)
{
  // Make sure that we don't go beyond t1
  if ( (t0 + K + DOLFIN_EPS) > t1 ) {
    K = t1 - t0;
    reached_endtime = true;
  }
  else
    t1 = t0 + K;
}
//-----------------------------------------------------------------------------
void TimeSlab::add(TimeSlab* timeslab)
{
  // Estimate the number of slabs
  //int n = pos + ceil_int( (t1 - timeslab->t0) / timeslab->length());
  
  // Make sure that the estimate is correct for the last time slab
  //if(timeslab->finished())
  //n = pos + 1;
  
  // Increase the size of the list if necessary
  //if ( n > timeslabs.size() )
  //timeslabs.resize(n);

  // Add the slab to the list
  //timeslabs(pos) = timeslab;
  timeslabs.push_back(timeslab);
}
//-----------------------------------------------------------------------------
