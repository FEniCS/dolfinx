// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Sparsity.h>
#include <dolfin/Component.h>
#include <dolfin/ODE.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabData.h>
#include <dolfin/RHS.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RHS::RHS(ODE& ode, TimeSlabData& data)
{
  // Save the ODE
  this->ode = &ode;

  // Save time slab data
  this->data = &data;

  // Initialize the solution vector
  u.init(ode.size());
  u = ode.u0;

}
//-----------------------------------------------------------------------------
RHS::~RHS()
{

}
//-----------------------------------------------------------------------------
real RHS::operator() (int index, int node, real t, TimeSlab* timeslab)
{
  // Update the solution vector
  update(index, node, t, timeslab);


  // Evaluate right hand side for current component
  return ode->f(u, t, index);
}
//-----------------------------------------------------------------------------
void RHS::update(int index, int node, real t, TimeSlab* timeslab)
{
  // Update the solution vector for all components that influence the
  // current component.

  dolfin_debug1("t: %lf", t);

  for (Sparsity::Iterator i(index, ode->sparsity); !i.end(); ++i)
  {
    u(i) = data->component(i)(node, t, timeslab);
    dolfin::cout << "u(" << i << "): " << u(i) << dolfin::endl;
  }
}
//-----------------------------------------------------------------------------
int RHS::size() const
{
  return ode->size();
}
//-----------------------------------------------------------------------------
