// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Sparsity.h>
#include <dolfin/Component.h>
#include <dolfin/ODE.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/ElementData.h>
#include <dolfin/RHS.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RHS::RHS(ODE& ode, ElementData& elmdata) : ode(ode), elmdata(elmdata)
{
  // Initialize the solution vector
  u.init(ode.size());
  u = ode.u0;
}
//-----------------------------------------------------------------------------
RHS::~RHS()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real RHS::operator() (unsigned int index, unsigned int node, real t, 
		      TimeSlab* timeslab)
{
  // Update the solution vector
  update(index, node, t, timeslab);

  // Evaluate right hand side for current component
  return ode.f(u, t, index);
}
//-----------------------------------------------------------------------------
void RHS::update(unsigned int index, unsigned int node, real t, 
		 TimeSlab* timeslab)
{
  // Update the solution vector for all components that influence the
  // current component.
  
  for (Sparsity::Iterator i(index, ode.sparsity); !i.end(); ++i)
    u(i) = elmdata.component(i)(node, t, timeslab);
}
//-----------------------------------------------------------------------------
unsigned int RHS::size() const
{
  return ode.size();
}
//-----------------------------------------------------------------------------
