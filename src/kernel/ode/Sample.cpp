// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSteppingData.h>
#include <dolfin/Sample.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Sample::Sample(TimeSteppingData& data, RHS& f, real t) :
  data(data), f(f), time(t)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Sample::~Sample()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int Sample::size() const
{
  return data.size();
}
//-----------------------------------------------------------------------------
real Sample::t() const
{
  return time;
}
//-----------------------------------------------------------------------------
real Sample::u(unsigned int index)
{
  //cout << endl;
  //cout << "Sample: ";
  //real u = data.u(index, time);
  //cout << endl;
  //return u;
  
  return data.u(index, time);
}
//-----------------------------------------------------------------------------
real Sample::k(unsigned int index)
{
  return data.k(index, time);
}
//-----------------------------------------------------------------------------
real Sample::r(unsigned int index)
{
  return data.r(index, time, f);
}
//-----------------------------------------------------------------------------
