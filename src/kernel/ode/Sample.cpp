// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/Sample.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Sample::Sample(TimeSlab& timeslab, real t,
	       std::string name, std::string label) : 
  Variable(name, label), timeslab(timeslab), time(t)
{
  // Prepare time slab for sample
  timeslab.sample(t);
}
//-----------------------------------------------------------------------------
Sample::~Sample()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int Sample::size() const
{
  return timeslab.size();
}
//-----------------------------------------------------------------------------
real Sample::t() const
{
  return time;
}
//-----------------------------------------------------------------------------
real Sample::u(unsigned int index)
{
  return timeslab.usample(index, time);
}
//-----------------------------------------------------------------------------
real Sample::k(unsigned int index)
{
  return timeslab.ksample(index, time);
}
//-----------------------------------------------------------------------------
real Sample::r(unsigned int index)
{
  return timeslab.rsample(index, time);
}
//-----------------------------------------------------------------------------
