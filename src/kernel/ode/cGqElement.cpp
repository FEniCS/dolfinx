// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/RHS.h>
#include <dolfin/cGqMethods.h>
#include <dolfin/cGqElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
cGqElement::cGqElement(unsigned int q, unsigned int index, real t0, real t1) :
  Element(q, index, t0, t1)
{  
  cG.init(q);
}
//-----------------------------------------------------------------------------
cGqElement::~cGqElement()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Element::Type cGqElement::type() const
{
  return Element::cg;
}
//-----------------------------------------------------------------------------
real cGqElement::value(real t) const
{
  // Special case: initial value
  if ( t == t0 )
    return values[0];

  real tau = (t - t0) / (t1 - t0);

  real sum = 0.0;
  for (unsigned int i = 0; i <= q; i++)
    sum += values[i] * cG(q).basis(i, tau);

  return sum;
}
//-----------------------------------------------------------------------------
real cGqElement::initval() const
{
  return values[0];
}
//-----------------------------------------------------------------------------
real cGqElement::dx() const
{
  real dudx = 0.0;

  for (unsigned int i = 0; i <= q; i++)
    dudx += values[i] * cG(q).derivative(i);

  return dudx / (t1 - t0);
}
//-----------------------------------------------------------------------------
void cGqElement::update(real u0)
{
  values[0] = u0;
}
//-----------------------------------------------------------------------------
void cGqElement::update(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Update nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i] = values[0] + integral(i);
}
//-----------------------------------------------------------------------------
void cGqElement::update(RHS& f, real alpha)
{
  // Evaluate right-hand side
  feval(f);

  // Compute weight for old value
  real w0 = 1.0 - alpha;

  // Update nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i] = w0*values[i] + alpha*(values[0] + integral(i));
}
//-----------------------------------------------------------------------------
void cGqElement::reset(real u0)
{
  for (unsigned int i = 0; i <= q; i++)
    values[i] = u0;
}
//-----------------------------------------------------------------------------
real cGqElement::computeTimeStep(real TOL, real r, real kmax) const
{
  // Compute new time step based on residual

  if ( abs(r) < DOLFIN_EPS )
    return kmax;

  // FIXME: Missing stability factor, interpolation constant, power

  //dolfin_debug1("k: %100.100lf", TOL / abs(r));
  //dolfin_debug1("r: %100.100lf", abs(r));

  return TOL / abs(r);
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

  real k = t1 - t0;

  for (unsigned int i = 0; i <= q; i++)
    this->f(i) = f(_index, i, t0 + cG(q).point(i)*k);
}
//-----------------------------------------------------------------------------
real cGqElement::integral(unsigned int i) const
{
  real k = t1 - t0;

  real sum = 0.0;
  for (unsigned int j = 0; j <= q; j++)
    sum += cG(q).weight(i, j) * f(j);

  return k * sum;
}
//-----------------------------------------------------------------------------
real cGqElement::computeDiscreteResidual(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  return (values[q] - values[0] - integral(q)) / timestep();
}
//-----------------------------------------------------------------------------
