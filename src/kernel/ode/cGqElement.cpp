// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/cGqMethods.h>
#include <dolfin/cGqElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
cGqElement::cGqElement(unsigned int q, unsigned int index, TimeSlab* timeslab) :
  Element(q, index, timeslab)
{  
  cG.init(q);

  //dolfin_debug1("cGqElement::ctor: %p", this);
}
//-----------------------------------------------------------------------------
real cGqElement::eval(real t) const
{
  //dolfin_debug1("t: %lf", t);
  
  //for(int i = 0; i < q + 1; i++)
  //  dolfin::cout << "values[" << i << "]: " << values[i] << dolfin::endl;

  real tau = (t - starttime()) / timestep();

  real sum = 0.0;
  for (unsigned int i = 0; i <= q; i++)
    sum += values[i] * cG(q).basis(i, tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
real cGqElement::dx() const
{
  real dudx = 0.0;

  for (unsigned int i = 0; i <= q; i++)
    dudx += values[i] * cG(q).derivative(i);

  return dudx;
}
//-----------------------------------------------------------------------------
void cGqElement::update(real u0)
{
  // FIXME: Maybe only the initial value should be updated, but the all
  // values updated the first time? Maybe the initial value must be
  // supplied to the constructor? Maybe the difference between the new
  // initial value and the previous should be added to all values?

  values[0] = u0;

  // Update nodal values
  //for (unsigned int i = 0; i <= q; i++)
  //values[i] = u0;
}
//-----------------------------------------------------------------------------
void cGqElement::update(RHS& f)
{
  //dolfin_debug1("Updating cG(%d) element", q);

  // Evaluate right-hand side
  feval(f);

  // Update nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i] = values[0] + integral(i);
}
//-----------------------------------------------------------------------------
real cGqElement::computeTimeStep(real r) const
{
  // Compute new time step based on residual and current time step

  // FIXME: Return maximum time step
  if ( abs(r) < DOLFIN_EPS )
    return 0.25;

  // FIXME: Use tolerance and stability factor
  real k0 = timestep();
  real k1 = 0.1 / r;

  return 2.0*k0*k1 / (k0 + k1);
}
//-----------------------------------------------------------------------------
void cGqElement::feval(RHS& f)
{
  // The right-hand side is evaluated once, before the nodal values
  // are updated, to avoid repeating the evaluation for each degree of
  // freedom. The local iterations are thus more of Jacobi type than
  // Gauss-Seidel.  This is probably more efficient, at least for
  // higher order methods (where we have more degrees of freedom) and
  // when function evaluations are expensive.

  real t0 = starttime();
  real k = timestep();

  for (unsigned int i = 0; i <= q; i++)
    this->f(i) = f(index, i, t0 + cG(q).point(i)*k, timeslab);
}
//-----------------------------------------------------------------------------
real cGqElement::integral(unsigned int i) const
{
  real k = timestep();

  real sum = 0.0;
  for (unsigned int j = 0; j <= q; j++)
    sum += cG(q).weight(i, j) * f(j);

  return k * sum;
}
//-----------------------------------------------------------------------------
