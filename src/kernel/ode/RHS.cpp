// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Sparsity.h>
#include <dolfin/ODE.h>
#include <dolfin/Solution.h>
#include <dolfin/Function.h>
#include <dolfin/RHS.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RHS::RHS(ODE& ode, Solution& solution) :
  ode(ode), solution(&solution), function(0), u(ode.size())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
RHS::RHS(ODE& ode, Function& function) :
  ode(ode), solution(0), function(&function), u(ode.size())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
RHS::~RHS()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int RHS::size() const
{
  return ode.size();
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
real RHS::dFdU(unsigned int i, unsigned int j, real t)
{
  // Update u(j) in case f(i) does not depend on u(j)
  u(j) = 0.0;

  // Update the solution vector
  update(i, t);
  
  // Small change in u_j
  double dU = DOLFIN_SQRT_EPS * abs(u(j));
  if ( dU == 0.0 )
    dU = DOLFIN_SQRT_EPS;

  // Save value of u_j
  real uj = u(j);
  
  // F values
  u(j) -= 0.5 * dU;
  real f1 = ode.f(u, t, i);
  
  u(j) = uj + 0.5*dU;
  real f2 = ode.f(u, t, i);
         
  // Compute derivative
  if ( abs(f1-f2) < DOLFIN_EPS * max(abs(f1), abs(f2)) )
    return 0.0;

  return (f2-f1) / dU;
}
//-----------------------------------------------------------------------------
void RHS::update(unsigned int index, unsigned int node, real t)
{
  // FIXME: Use nodal values if possible
  update(index, t);
}
//-----------------------------------------------------------------------------
void RHS::update(unsigned int index, real t)
{
  // Update the solution vector for all components that influence the
  // current component.

  // Maybe Solution and Function should be closer related (or we should
  // not use Solution), to avoid having two different versions.

  if ( solution != 0 )
  {
    dolfin_assert(solution);
    for (Sparsity::Iterator i(index, ode.sparsity); !i.end(); ++i) 
      u(i) = (*solution)(i, t);
  }
  else
  {
    dolfin_assert(function);
    for (Sparsity::Iterator i(index, ode.sparsity); !i.end(); ++i) 
      u(i) = (*function)(i, t);
  }
}
//-----------------------------------------------------------------------------
