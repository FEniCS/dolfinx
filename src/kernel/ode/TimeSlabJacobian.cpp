// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/NewMethod.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/TimeSlabJacobian.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabJacobian::TimeSlabJacobian(NewTimeSlab& timeslab)
  : ode(timeslab.ode), method(*timeslab.method), Jvalues(0), Jindices(0)
{
  // Allocate Jacobian row indices
  Jindices = new uint[ode.size()];
  
  // Compute start of each row
  uint sum = 0;
  for (uint i = 0; i < ode.size(); i++)
  {
    Jindices[i] = sum;
    sum += ode.dependencies[i].size();
  }

  // Allocate Jacobian values
  Jvalues = new real[sum];
  for (uint pos = 0; pos < sum; pos++)
    Jvalues[pos] = 0.0;

  dolfin_info("Generated Jacobian data structure for %d dependencies.", sum);
}
//-----------------------------------------------------------------------------
TimeSlabJacobian::~TimeSlabJacobian()
{
  if ( Jvalues ) delete [] Jvalues;
  if ( Jindices ) delete [] Jindices;
}
//-----------------------------------------------------------------------------
void TimeSlabJacobian::update(const NewTimeSlab& timeslab)
{
  // Compute Jacobian at the beginning of the slab
  real t = timeslab.starttime();
  dolfin_info("Recomputing Jacobian matrix at t = %f.", t);
  
  // Compute Jacobian
  for (uint i = 0; i < ode.size(); i++)
  {
    const NewArray<uint>& deps = ode.dependencies[i];
    for (uint pos = 0; pos < deps.size(); pos++)
      Jvalues[Jindices[i] + pos] = ode.dfdu(timeslab.u0, t, i, deps[pos]);
  }
}
//-----------------------------------------------------------------------------
