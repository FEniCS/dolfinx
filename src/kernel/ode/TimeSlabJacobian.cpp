// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/NewMethod.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/TimeSlabJacobian.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabJacobian::TimeSlabJacobian(NewTimeSlab& timeslab)
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
