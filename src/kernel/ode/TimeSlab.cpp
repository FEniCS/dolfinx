// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>

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
void TimeSlab::update(RHS& f, TimeSlabData& data)
{
  dolfin_info("Updating time slab");

  // First update the time slabs
  updateTimeSlabs(f, data);

  // Then update the elements
  updateElements(f, data);
}
//-----------------------------------------------------------------------------
bool TimeSlab::within(real t) const
{
  // Check if t is in the interval (t0,t1] = [t0 + eps, t1 + eps].
  // We need to make sure that we include the end-point. Otherwise
  // a round-off error may cause the next interval to be chosen,
  // which is not what we want, at least not for dG.

  //t -= DOLFIN_EPS;

  //return (t0 <= t) && (t <= t1);
  return (t0 < t) && (t <= t1);
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
  partition.update(data, offset);

  // Compute a partition into small and large time steps
  partition.partition(offset, end, K);

  dolfin_debug1("offset: %d", offset);
  dolfin_debug1("end: %d", end);

  dolfin_debug1("Adjusting time step K = %f", K);

  // Adjust and set the size of this time slab 
  setsize(K);

  dolfin_debug1("New end time = %f", t1);

  if(end < f.size())
  {
    dolfin_debug("Create subslabs");

    // Create time slabs for the components with small time steps
    createTimeSlabs(f, data, partition, end);
  }

  dolfin_debug1("offset: %d", offset);
  dolfin_debug1("end: %d", end);

  // Create elements for the components with large time steps
  createElements(f, data, partition, offset, end);
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
			      Partition& partition, int offset, int end)
{
  // Instead of storing a list of elements, we store two iterators to
  // the list of elements in TimeSlabData.

  dolfin_debug1("offset: %d", offset);
  dolfin_debug1("end: %d", end);


  dolfin_info("Creating elements");

  // FIXME: choose element and order here
  Element::Type type = Element::cg;
  //Element::Type type = Element::dg;

  int q = 1;
  //int q = 0;

  dolfin_debug1("slab: %p", this);
  
  // Create elements
  for (int i = offset; i < end; i++) {
    Element* element = data.createElement(type, q, partition.index(i), this);
    elements.push_back(element);
  }

  dolfin_debug("Update elements:");

  // Update elements
  updateElements(f, data);
}
//-----------------------------------------------------------------------------
void TimeSlab::updateTimeSlabs(RHS& f, TimeSlabData& data)
{
  for (int i = 0; i < timeslabs.size(); i++)
  {
    //if(i > 0)
    //{
    //  timeslabs[i]->updateu0(*(timeslabs[i - 1]));
    //}

    timeslabs[i]->update(f, data);
  }
}
//-----------------------------------------------------------------------------
void TimeSlab::updateElements(RHS& f, TimeSlabData& data)
{
  dolfin_debug1("elements: %d", elements.size());
  dolfin_debug2("timeslab: %lf-%lf", starttime(), endtime());

  updateu0(data);


  for(int i = 0; i < 3; i++)
  {
    dolfin::cout << "iteration: " << i << dolfin::endl;
    
    for(std::vector<Element *>::iterator it = elements.begin();
	it != elements.end(); it++)
    {
      Element *e = *it;
      
      //if(e->starttime() == data
      // Update u0 (from the end values of previous slabs)
      //Component &c = data.component(e->index);
      //real u0 = c(e->starttime());
      
      //e->update(u0);
      
      /// Iterate over element
      
      
      e->update(f);
      real value, residual;
      
      residual = e->computeResidual(f);
      
      value = e->eval(e->starttime());
      dolfin::cout << "element value at starttime: " << value << dolfin::endl;
      value = e->eval(e->endtime());
      dolfin::cout << "element value at endtime: " << value << dolfin::endl;
      dolfin::cout << "element residual at endtime: " <<
	residual << dolfin::endl;
    }
  }
}
//-----------------------------------------------------------------------------
void TimeSlab::setsize(real K)
{
  real t0K = t0 + K;


  /*
  dolfin_debug("K:");
  dolfin_debug1("%.200lf", K);
  dolfin_debug("t0:");
  dolfin_debug1("%.200lf", t0);
  dolfin_debug("t1:");
  dolfin_debug1("%.200lf", t1);
  dolfin_debug("t0K:");
  dolfin_debug1("%.200lf", t0K);
  dolfin_debug("t0K - t1:");
  dolfin_debug1("%.200lf", t0K - t1);
  dolfin_debug("t0K >= t1:");
  dolfin_debug1("%d", t0K >= t1);
  */



  // Make sure that we don't go beyond t1
  //if ( (t0 + K + DOLFIN_EPS) > t1 ) {
  if(t0K >= t1 ) {
    K = t1 - t0;
    t1 = t0 + K;
    reached_endtime = true;
  }
  else
  {
    //dolfin_debug("t1:");
    //dolfin_debug1("%.200lf", t1);
    //dolfin_debug("t0K:");
    //dolfin_debug1("%.200lf", t0K);
    //dolfin_debug("t0K - t1:");
    //dolfin_debug1("%.200lf", t0K - t1);
    

    t1 = t0 + K;
  }
  //dolfin_debug1("re: %d", reached_endtime);

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
void TimeSlab::updateu0(TimeSlabData &data)
{
  for(int i = 0; i < elements.size(); i++)
  {
    Element *e = elements[i];
    //Element *preve = prevslab.elements[i];

    real u0 = data.component(e->index)(e->starttime());

    dolfin_debug3("u0(%d, %lf): %lf", e->index, e->starttime(), u0);

    //real u0 = preve->eval(preve->endtime());
    e->update(u0);
  }
}
//-----------------------------------------------------------------------------

