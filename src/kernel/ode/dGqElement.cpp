// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/dGqMethods.h>
#include <dolfin/dGqElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dGqElement::dGqElement(unsigned int q, unsigned int index, TimeSlab* timeslab) : 
  Element(q, index, timeslab)
{  
  dG.init(q);
}
//-----------------------------------------------------------------------------
real dGqElement::value(real t) const
{
  real tau = (t - starttime()) / timestep();

  real sum = 0.0;
  for (unsigned int i = 0; i <= q; i++)
    sum += values[i] * dG(q).basis(i, tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
real dGqElement::dx() const
{
  real dudx = 0.0;

  for (unsigned int i = 0; i <= q; i++)
    dudx += values[i] * dG(q).derivative(i);

  return dudx;
}
//-----------------------------------------------------------------------------
void dGqElement::update(real u0)
{
  // FIXME: See comment on dGqElement::update()

  // Update initial value
  this->u0 = u0;

  // Update nodal values
  for (unsigned int i = 0; i <= q; i++)
    values[i] = u0;
}
//-----------------------------------------------------------------------------
void dGqElement::update(RHS& f)
{
  dolfin_debug4("Updating dG(%d) element: %d, %lf-%lf",
		q, _index, starttime(), endtime());

  dolfin::cout << "values[0]: " << values[0] << dolfin::endl;

  // Evaluate right-hand side
  feval(f);

  // Update nodal values
  for (unsigned int i = 0; i <= q; i++)
    values[i] = u0 + integral(i);
    //values[i] = values[0] + integral(i);
}
//-----------------------------------------------------------------------------
real dGqElement::computeTimeStep(real r) const
{
  // Compute new time step based on residual

  // FIXME: Return maximum time step
  if ( abs(r) < DOLFIN_EPS )
    return 0.25;

  return 0.1 / abs(r);
}
//-----------------------------------------------------------------------------
void dGqElement::feval(RHS& f)
{
  // See comment on cGqElement::feval()

  real t0 = starttime();
  real k = timestep();
  
  for (unsigned int i = 0; i <= q; i++)
    this->f(i) = f(_index, i, t0 + dG(q).point(i)*k, timeslab);
}
//-----------------------------------------------------------------------------
real dGqElement::integral(unsigned int i) const
{
  real k = timestep();

  real sum = 0.0;
  for (unsigned int j = 0; j <= q; j++)
    sum += dG(q).weight(i, j) * f(j);

  return k * sum;
}
//-----------------------------------------------------------------------------
