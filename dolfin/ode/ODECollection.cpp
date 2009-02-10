// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-02-09
// Last changed: 2009-02-09

#include "ODECollection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ODECollection::ODECollection(ODE& ode, uint n)
  : ode(ode), u(ode), n(n), states(0)
{
  message("Creating ODE collection of size %d x %d.", n, ode.size());

  // Allocate state vectors
  states = new real[n*ode.size()];
}
//-----------------------------------------------------------------------------
ODECollection::~ODECollection()
{
  delete [] states;
}
//-----------------------------------------------------------------------------
void ODECollection::solve(real t0, real t1)
{
  begin("Solving ODE collection on interval [%g, %g].", t0, t1);

  // Iterate over all ODE systems
  for (uint i = 0; i < n; i++)
  {
    begin("Time-stepping ODE number %d.", i);
    
    // Compute offset for ODE
    const uint offset = i*ode.size();

    // Copy initial state from state vector
    ode.set_state(states + offset);
    
    // Time-stepping
    ode.solve(u, t0, t1);

    // Copy final state to state vector
    ode.get_state(states + offset);

    end();
  }

  end();
}
//-----------------------------------------------------------------------------
void ODECollection::set_state(uint i, const real* u)
{
  const uint offset = i*ode.size();
  for (uint j = 0; j < ode.size(); j++)
    states[offset + j] = u[j];
}
//-----------------------------------------------------------------------------
void ODECollection::set_state(const real* u)
{
  for (uint j = 0; j < n*ode.size(); j++)
    states[j] = u[j];
}
//-----------------------------------------------------------------------------
void ODECollection::get_state(uint i, real* u)
{
  const uint offset = i*ode.size();
  for (uint j = 0; j < ode.size(); j++)
    u[j] = states[offset + j];
}
//-----------------------------------------------------------------------------
void ODECollection::get_state(real* u)
{
  for (uint j = 0; j < n*ode.size(); j++)
    u[j] = states[j];
}
//-----------------------------------------------------------------------------
