// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/NewArray.h>
#include <dolfin/Sparsity.h>
#include <dolfin/ODE.h>
#include <dolfin/Solution.h>
#include <dolfin/Function.h>
#include <dolfin/RHS.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
RHS::RHS(ODE& ode, Solution& solution) :
  N(ode.size()), ode(ode), solution(&solution), function(0), u(ode.size()),
  illegal_number("Warning: Right-hand side returned illegal number (nan or inf).", 3)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
RHS::RHS(ODE& ode, Function& function) :
  N(ode.size()), ode(ode), solution(0), function(&function), u(ode.size()),
  illegal_number("Warning: Right-hand side returned illegal number (nan or inf).", 3)
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
  return check(ode.f(u, t, index));
}
//-----------------------------------------------------------------------------
real RHS::dfdu(unsigned int i, unsigned int j, real t)
{
  // Update u(j) in case f(i) does not depend on u(j)
  u(j) = 0.0;

  // Update the solution vector, take node number to zero, since we have
  // to take something.
  update(i, 0, t);
  
  // Small change in u_j
  real dU = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * abs(u(j)));

  // Save value of u_j
  real uj = u(j);
  
  // Compute F values
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
  // Update the solution vector for all components that influence the
  // current component. Maybe Solution and Function should be closer
  // related (or we should not use Solution), to avoid having two
  // different versions.

  if ( solution )
    updateSolution(index, node, t);
  else
    updateFunction(index, t);
}
//-----------------------------------------------------------------------------
void RHS::updateSolution(unsigned int index, unsigned int node, real t)
{
  dolfin_assert(solution);
  
  if ( ode.sparsity.sparse() )
  {
    const NewArray<unsigned int>& row = ode.sparsity.row(index);
    for (unsigned int pos = 0; pos < row.size(); ++pos)
      u(row[pos]) = solution->u(row[pos], node, t);
  }
  else
  {
    for (unsigned int j = 0; j < N; j++)
      u(j) = solution->u(j, node, t);
  }
}
//-----------------------------------------------------------------------------
void RHS::updateFunction(unsigned int index, real t)
{
  dolfin_assert(function);

  if ( ode.sparsity.sparse() )
  {
    const NewArray<unsigned int>& row = ode.sparsity.row(index);
    for (unsigned int pos = 0; pos < row.size(); ++pos)
      u(row[pos]) = (*function)(row[pos], t);
  }
  else
  {
    for (unsigned int j = 0; j < N; j++)
      u(j) = (*function)(j, t);
  }
}
//-----------------------------------------------------------------------------
