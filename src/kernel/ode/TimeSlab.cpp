// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSteppingData.h>
#include <dolfin/TimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlab::TimeSlab(real t0, real t1)
{
  this->t0 = t0;
  this->t1 = t1;
  reached_endtime = false;
}
//-----------------------------------------------------------------------------
TimeSlab::~TimeSlab()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool TimeSlab::within(real t) const
{
  // Check if t is in the interval (t0,t1] = [t0 + eps, t1 + eps].
  // We need to make sure that we include the end-point. Otherwise
  // a round-off error may cause the next interval to be chosen,
  // which is not what we want, at least not for dG.

  // FIXME: Is this necessary?
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
void TimeSlab::setsize(real K, const TimeSteppingData& data)
{
  // Make sure that we don't go beyond t1

  if ( K > data.threshold() * (t1 - t0) )
  {
    K = t1 - t0;
    t1 = t0 + K;
    reached_endtime = true;
  }
  else
    t1 = t0 + K;
}
//-----------------------------------------------------------------------------
void TimeSlab::updateElements(RHS& f, TimeSteppingData& data)
{
  // Update initial values
  updateu0(data);

  // Update elements
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    // Get the element
    Element* element = elements[i];
    
    // Update element
    element->update(f);
    
    // Write debug info
    data.debug(*element, TimeSteppingData::update);
  }
}
//-----------------------------------------------------------------------------
void TimeSlab::updateu0(TimeSteppingData& data)
{
  // Update initial values
  for (unsigned int i = 0; i < elements.size(); i++)
  {
    Element* e = elements[i];
    real u0 = data.u(e->index(), e->starttime());
    e->update(u0);
  }  
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const TimeSlab& timeslab)
{
  stream << "[ TimeSlab of length " << timeslab.length()
	 << " between t0 = " << timeslab.starttime()
	 << " and t1 = " << timeslab.endtime() << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
