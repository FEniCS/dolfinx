// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/dGqMethods.h>
#include <dolfin/dGqElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dGqElement::dGqElement(int q, int index, TimeSlab* timeslab) : 
  Element(q, index, timeslab)
{  
  dG.init(q);
}
//-----------------------------------------------------------------------------
void dGqElement::update(real u0)
{
  this->u0 = u0;
}
//-----------------------------------------------------------------------------
real dGqElement::eval(real t) const
{
  real tau = 0.0;

  real sum = 0.0;
  for (int i = 0; i <= q; i++)
    sum += values[i] * dG(q).basis(i,tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
real dGqElement::eval(int node) const
{
  dolfin_assert(node >= 0);
  dolfin_assert(node <= q);
  
  return values[node];
}
//-----------------------------------------------------------------------------
void dGqElement::update(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Update nodal values
  for (int i = 0; i <= q; i++)
    values[i] = u0 + integral(i);
}
//-----------------------------------------------------------------------------
real dGqElement::newTimeStep() const
{
  // Compute new time step based on residual and current time step
  
  
  // Not implemented, return a random time step
  return dolfin::rand();
}
//-----------------------------------------------------------------------------
void dGqElement::feval(RHS& f)
{
  // See comment on cGqElement::feval()

  real t0 = 0.0;
  real k = 0.1;
  
 for (int i = 0; i <= q; i++)
    this->f(i) = f(index, i, t0 + dG(q).point(i)*k, timeslab);
}
//-----------------------------------------------------------------------------
real dGqElement::integral(int i) const
{
  real k = 0.1;

  real sum = 0.0;
  for (int j = 0; j <= q; j++)
    sum += dG(q).weight(i,j) * f(j);

  return k * sum;
}
//-----------------------------------------------------------------------------
