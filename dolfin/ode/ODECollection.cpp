// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2009
//
// First added:  2009-02-09
// Last changed: 2009-09-10

#include "ODECollection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ODECollection::ODECollection(ODE& ode, uint num_systems)
  : ode(ode), num_systems(num_systems), states(new real[num_systems*ode.size()])
{
  info(TRACE, "Creating ODE collection of size %d x %d.", num_systems, ode.size());

}
//-----------------------------------------------------------------------------
ODECollection::~ODECollection()
{
  delete [] states;
}
//-----------------------------------------------------------------------------
void ODECollection::solve(real t0, real t1)
{
  begin("Solving ODE collection on interval [%g, %g].",
        to_double(t0), to_double(t1));

  // Iterate over all ODE systems
  for (uint system = 0; system < num_systems; system++)
  {
    begin("Time-stepping ODE system number %d.", system);

    // Compute offset for ODE
    const uint offset = system*ode.size();

    // Wrap relevant part of state array in an Array<real>
    Array<real> s(ode.size(), states+offset);

    // Copy initial state from state vector
    ode.set_state(s);

    // Call user-defined update
    update(s, t0, system);

    // Time-stepping
    ode.solve(t0, t1);

    // Copy final state to state vector
    ode.get_state(s);

    end();
  }

  end();
}
//-----------------------------------------------------------------------------
void ODECollection::set_state(uint system, const Array<real>& u)
{
  const uint offset = system*ode.size();
  for (uint j = 0; j < ode.size(); j++)
    states[offset + j] = u[j];
}
//-----------------------------------------------------------------------------
void ODECollection::set_state(const Array<real>& u)
{
  for (uint j = 0; j < num_systems*ode.size(); j++)
    states[j] = u[j];
}
//-----------------------------------------------------------------------------
void ODECollection::get_state(uint system, Array<real>& u)
{
  const uint offset = system*ode.size();
  for (uint j = 0; j < ode.size(); j++)
    u[j] = states[offset + j];
}
//-----------------------------------------------------------------------------
void ODECollection::get_state(Array<real>& u)
{
  for (uint j = 0; j < num_systems*ode.size(); j++)
    u[j] = states[j];
}
//-----------------------------------------------------------------------------
void ODECollection::update(Array<real>& u, real t, uint system)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
