// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Sparsity.h>
#include <dolfin/ODE.h>
#include <dolfin/Solution.h>
#include <dolfin/RHS.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RHS::RHS(ODE& ode, Solution& solution) :
  ode(ode), solution(solution), u(ode.size())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
RHS::~RHS()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real RHS::operator() (unsigned int index, unsigned int node, real t)
{
  // Update the solution vector
  update(index, node, t);
  
  // Evaluate right hand side for current component
  return ode.f(u, t, index);
}
//-----------------------------------------------------------------------------
void RHS::update(unsigned int index, unsigned int node, real t)
{
  // Update the solution vector for all components that influence the
  // current component.
  
  // FIXME: Use nodal values if possible

  for (Sparsity::Iterator i(index, ode.sparsity); !i.end(); ++i) 
    u(i) = solution(i,t);
}
//-----------------------------------------------------------------------------
unsigned int RHS::size() const
{
  return ode.size();
}
//-----------------------------------------------------------------------------
