// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/TimeSlabSample.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabSample::TimeSlabSample(TimeSlab& timeslab, TimeSlabData& data, 
			       RHS& f, real t) :
  timeslab(timeslab), data(data), f(f), t(t)
{
  // Check that the given time is within the interval

  if ( t < timeslab.starttime() || t > timeslab.endtime() )
    dolfin_error("Sample point must be within the time slab.");
}
//-----------------------------------------------------------------------------
TimeSlabSample::~TimeSlabSample()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int TimeSlabSample::size() const
{
  return data.size();
}
//-----------------------------------------------------------------------------
real TimeSlabSample::time() const
{
  return t;
}
//-----------------------------------------------------------------------------
real TimeSlabSample::value(unsigned int index)
{
  return data.component(index)(t);
}
//-----------------------------------------------------------------------------
real TimeSlabSample::timestep(unsigned int index)
{
  return data.component(index).element(t).timestep();
}
//-----------------------------------------------------------------------------
real TimeSlabSample::residual(unsigned int index)
{
  return data.component(index).element(t).computeResidual(f);
}
//-----------------------------------------------------------------------------
