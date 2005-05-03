// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabJacobian.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabJacobian::TimeSlabJacobian(TimeSlab& timeslab)
  : ode(timeslab.ode), method(*timeslab.method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
TimeSlabJacobian::~TimeSlabJacobian()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TimeSlabJacobian::update()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
