// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
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
unsigned int cGqElement::size() const
{
  return q;
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
real cGqElement::value(unsigned int node, real t) const
{
  // First check if the node matches
  if ( t0 + cG(q).point(node)*timestep() == t )
    return values[node];

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
real cGqElement::update(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Save old value
  real u1 = values[q];

  // Update nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i] = values[0] + integral(i);

  // Return increment
  return values[q] - u1;
}
//-----------------------------------------------------------------------------
real cGqElement::update(RHS& f, real* values)
{
  // Evaluate right-hand side
  feval(f);

  // Update nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i-1] = this->values[0] + integral(i);

  // Return increment
  return values[q] - this->values[q];
}
//-----------------------------------------------------------------------------
real cGqElement::update(RHS& f, real alpha)
{
  // Evaluate right-hand side
  feval(f);

  // Compute weight for old value
  real w0 = 1.0 - alpha;

  // Save old value
  real u1 = values[q];

  // Update nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i] = w0*values[i] + alpha*(values[0] + integral(i));

  // Return increment
  return values[q] - u1;
}
//-----------------------------------------------------------------------------
real cGqElement::update(RHS& f, real alpha, real* values)
{
  // Evaluate right-hand side
  feval(f);

  // Compute weight for old value
  real w0 = 1.0 - alpha;

  // Update nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i-1] = w0*this->values[i] + alpha*(this->values[0] + integral(i));

  // Return increment
  return values[q] - this->values[q];
}
//-----------------------------------------------------------------------------
real cGqElement::updateLocalNewton(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Save old value
  real u1 = values[q];
  
  // Compute increments for nodal values
  for (unsigned int i = 1; i <= q; i++)
    b[q-1](i-1) = values[i] - (values[0] + integral(i));
  
  // Compute local Jacobian
  real dfdu = f.dfdu(_index, _index, endtime());
  real k = timestep();
  for (unsigned int i = 0; i < q; i++)
  {
    for (unsigned int j = 0; j < q; j++)
    {
      if ( i == j )
	A[q-1](i, j) = 1.0 - k*dfdu*cG(q).weight(i+1, j);
      else
	A[q-1](i, j) = - k*dfdu*cG(q).weight(i+1, j);
    }
  }

  // Solve linear system
  A[q-1].solve(x[q-1], b[q-1]);

  // Compute nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i] -= x[q-1](i-1);

  // Return increment
  return values[q] - u1;
}
//-----------------------------------------------------------------------------
real cGqElement::updateLocalNewton(RHS& f, real* values)
{
  // Evaluate right-hand side
  feval(f);
  
  // Compute increments for nodal values
  for (unsigned int i = 1; i <= q; i++)
    b[q-1](i-1) = this->values[i] - this->values[0] + integral(i);

  // Compute local Jacobian
  real dfdu = f.dfdu(_index, _index, endtime());
  real k = timestep();
  for (unsigned int i = 0; i < q; i++)
  {
    for (unsigned int j = 0; j < q; j++)
    {
      if ( i == j )
	A[q-1](i, j) = 1.0 - k*dfdu*cG(q).weight(i+1, j);
      else
	A[q-1](i, j) = - k*dfdu*cG(q).weight(i+1, j);
    }
  }
  
  // Solve linear system
  A[q-1].solve(x[q-1], b[q-1]);

  // Compute nodal values
  for (unsigned int i = 1; i <= q; i++)
    values[i] = this->values[i] - x[q-1](i-1);

  // Return increment
  return values[q] - this->values[q];
}
//-----------------------------------------------------------------------------
void cGqElement::set(real u0)
{
  for (unsigned int i = 0; i <= q; i++)
    values[i] = u0;
}
//-----------------------------------------------------------------------------
void cGqElement::set(const real* const values)
{
  for (unsigned int i = 1; i <= q; i++)
    this->values[i] = values[i-1];
}
//-----------------------------------------------------------------------------
void cGqElement::get(real* const values) const
{
  for (unsigned int i = 1; i <= q; i++)
    values[i-1] = this->values[i];
}
//-----------------------------------------------------------------------------
bool cGqElement::accept(real TOL, real r)
{
  real error = pow(timestep(), static_cast<real>(q)) * fabs(r);
  return error <= TOL;
}
//-----------------------------------------------------------------------------
real cGqElement::computeTimeStep(real TOL, real r, real kmax) const
{
  // Compute new time step based on residual

  if ( fabs(r) < DOLFIN_EPS )
    return kmax;

  // FIXME: Missing stability factor and interpolation constant
  return pow(TOL / fabs(r), 1.0 / static_cast<real>(q));
}
//-----------------------------------------------------------------------------
real cGqElement::computeDiscreteResidual(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Compute discrete residual
  return (values[q] - values[0] - integral(q)) / timestep();
}
//-----------------------------------------------------------------------------
real cGqElement::computeElementResidual(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Compute element residual
  return values[q] - values[0] - integral(q);
}
//-----------------------------------------------------------------------------
void cGqElement::feval(RHS& f)
{
  // The right-hand side is evaluated once, before the nodal values
  // are updated, to avoid repeating the evaluation for each degree of
  // freedom. The local iterations are thus more of Jacobi type than
  // Gauss-Seidel. This is probably more efficient, at least for
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
