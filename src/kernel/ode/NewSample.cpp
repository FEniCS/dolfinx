// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/NewSample.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewSample::NewSample(NewTimeSlab& timeslab, real t,
		     std::string name, std::string label) : 
  Variable(name, label), timeslab(timeslab), time(t)
{
  // Prepare time slab for sample
  timeslab.sample(t);
}
//-----------------------------------------------------------------------------
NewSample::~NewSample()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int NewSample::size() const
{
  return timeslab.size();
}
//-----------------------------------------------------------------------------
real NewSample::t() const
{
  return time;
}
//-----------------------------------------------------------------------------
real NewSample::u(unsigned int index)
{
  return timeslab.usample(index, time);
}
//-----------------------------------------------------------------------------
real NewSample::k(unsigned int index)
{
  return timeslab.ksample(index, time);
}
//-----------------------------------------------------------------------------
real NewSample::r(unsigned int index)
{
  return timeslab.rsample(index, time);
}
//-----------------------------------------------------------------------------
