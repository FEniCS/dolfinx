// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/cGqMethods.h>
#include <dolfin/cGqElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
cGqElement::cGqElement(int q, int index, int pos, TimeSlab* timeslab) : 
  GenericElement(q, index, pos, timeslab)
{  
  cG.init(q);
}
//-----------------------------------------------------------------------------
real cGqElement::eval(real t) const
{
  real tau = 0.0;

  real sum = 0.0;
  for (int i = 0; i <= q; i++)
    sum += values[i] * cG(q).basis(i,tau);
}
//-----------------------------------------------------------------------------
real cGqElement::eval(int node) const
{
  dolfin_assert(node >= 0);
  dolfin_assert(node <= q);
  
  return values[node];
}
//-----------------------------------------------------------------------------
void cGqElement::update(real u0)
{
  values[0] = u0;
}
//-----------------------------------------------------------------------------
void cGqElement::update(RHS& f)
{
  dolfin_debug1("Updating cG(%d) element", q);

  // Evaluate right-hand side
  feval(f);

  // Update nodal values
  for (int i = 1; i <= q; i++)
    values[i] = values[0] + integral(i);
}
//-----------------------------------------------------------------------------
real cGqElement::newTimeStep() const
{
  // Compute new time step based on residual and current time step

  // Not implemented, return a random time step
  return dolfin::rand();
}
//-----------------------------------------------------------------------------
void cGqElement::feval(RHS& f)
{
  // The right-hand side is evaluated once, before the nodal values
  // are updated, to avoid repeating the evaluation for each degree of
  // freedom.  The local iterations are thus more of Jacobi type than
  // Gauss-Seidel.  This is probably more efficient, at least for
  // higher order methods (where we have more degrees of freedom) and
  // when function evaluations are expensive.

  real t0 = 0.0;
  real k = 0.1;

  for (int i = 0; i <= q; i++)
    this->f(i) = f(index, i, t0 + cG(q).point(i)*k, timeslab);
}
//-----------------------------------------------------------------------------
real cGqElement::integral(int i) const
{
  real k = 0.1;
  real t0 = 0.0;

  real sum = 0.0;
  for (int j = 0; j <= q; j++)
    sum += cG(q).weight(i,j) * f(j);

  return k * sum;
}
//-----------------------------------------------------------------------------
