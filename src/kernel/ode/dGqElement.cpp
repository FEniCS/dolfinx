// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/RHS.h>
#include <dolfin/dGqMethods.h>
#include <dolfin/dGqElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dGqElement::dGqElement(unsigned int q, unsigned int index, real t0, real t1) : 
  Element(q, index, t0, t1)
{  
  dG.init(q);
}
//-----------------------------------------------------------------------------
dGqElement::~dGqElement()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Element::Type dGqElement::type() const
{
  return Element::dg;
}
//-----------------------------------------------------------------------------
unsigned int dGqElement::size() const
{
  return q + 1;
}
//-----------------------------------------------------------------------------
real dGqElement::value(real t) const
{
  // Special case: initial value
  if ( t == t0 )
    return u0;

  real tau = (t - t0) / (t1 - t0);

  real sum = 0.0;
  for (unsigned int i = 0; i <= q; i++)
    sum += values[i] * dG(q).basis(i, tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
real dGqElement::value(unsigned int node, real t) const
{
  // First check if the node matches
  if ( t0 + dG(q).point(node)*timestep() == t )
    return values[node];

  // Special case: initial value
  if ( t == t0 )
    return values[0];

  real tau = (t - t0) / (t1 - t0);

  real sum = 0.0;
  for (unsigned int i = 0; i <= q; i++)
    sum += values[i] * dG(q).basis(i, tau);

  return sum;
}
//-----------------------------------------------------------------------------
real dGqElement::initval() const
{
  return u0;
}
//-----------------------------------------------------------------------------
real dGqElement::dx() const
{
  real dudx = 0.0;

  for (unsigned int i = 0; i <= q; i++)
    dudx += values[i] * dG(q).derivative(i);

  return dudx / (t1 - t0);
}
//-----------------------------------------------------------------------------
void dGqElement::update(real u0)
{
  this->u0 = u0;
}
//-----------------------------------------------------------------------------
real dGqElement::update(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Save old value
  real u1 = values[q];
  
  // Update nodal values
  for (unsigned int i = 0; i <= q; i++)
    values[i] = u0 + integral(i);

  // Return increment
  return values[q] - u1;
}
//-----------------------------------------------------------------------------
real dGqElement::update(RHS& f, real* values)
{
  // Evaluate right-hand side
  feval(f);

  // Update nodal values
  for (unsigned int i = 0; i <= q; i++)
    values[i] = u0 + integral(i);
  
  // Return increment
  return values[q] - this->values[q];
}
//-----------------------------------------------------------------------------
real dGqElement::update(RHS& f, real alpha)
{
  // Evaluate right-hand side
  feval(f);

  // Compute weight for old value
  real w0 = 1.0 - alpha;
 
  // Save old value
  real u1 = values[q];

  // Update nodal values
  for (unsigned int i = 0; i <= q; i++)
    values[i] = w0*values[i] + alpha*(u0 + integral(i));

  // Return increment
  return values[q] - u1;
}
//-----------------------------------------------------------------------------
real dGqElement::update(RHS& f, real alpha, real* values)
{
  // Evaluate right-hand side
  feval(f);

  // Compute weight for old value
  real w0 = 1.0 - alpha;

  // Update nodal values
  for (unsigned int i = 0; i <= q; i++)
    values[i] = w0*this->values[i] + alpha*(u0 + integral(i));

  // Return increment
  return values[q] - this->values[q];
}
//-----------------------------------------------------------------------------
real dGqElement::updateLocalNewton(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Save old value
  real u1 = values[q];
  
  // Compute increments for nodal values
  for (unsigned int i = 0; i <= q; i++)
    b[q](i) = values[i] - (u0 + integral(i));
  
  // Compute local Jacobian
  real dfdu = f.dfdu(_index, _index, endtime());
  real k = timestep();
  for (unsigned int i = 0; i <= q; i++)
  {
    for (unsigned int j = 0; j <= q; j++)
    {
      if ( i == j )
	A[q](i, j) = 1.0 - k*dfdu*dG(q).weight(i, j);
      else
	A[q](i, j) = - k*dfdu*dG(q).weight(i, j);
    }
  }

  // Solve linear system
  A[q].solve(x[q], b[q]);

  // Compute nodal values
  for (unsigned int i = 0; i <= q; i++)
    values[i] -= x[q](i);

  // Return increment
  return values[q] - u1;
}
//-----------------------------------------------------------------------------
real dGqElement::updateLocalNewton(RHS& f, real* values)
{
  // Evaluate right-hand side
  feval(f);

  // Compute new nodal values
  for (unsigned int i = 0; i <= q; i++)
    b[q](i) = u0 + integral(i);
  
  // Compute local Jacobian
  real dfdu = f.dfdu(_index, _index, endtime());
  real k = timestep();
  for (unsigned int i = 0; i <= q; i++)
  {
    for (unsigned int j = 0; j <= q; j++)
    {
      if ( i == j )
	A[q](i, j) = 1.0 - k*dfdu*dG(q).weight(i, j);
      else
	A[q](i, j) = - k*dfdu*dG(q).weight(i, j);
    }
  }

  // Solve linear system
  A[q].solve(x[q], b[q]);

  // Compute nodal values
  for (unsigned int i = 0; i <= q; i++)
    values[i] = this->values[i] - x[q](i);

  // Return increment
  return values[q] - this->values[q];
}
//-----------------------------------------------------------------------------
void dGqElement::set(real u0)
{
  this->u0 = u0;
  for (unsigned int i = 0; i <= q; i++)
    values[i] = u0;
}
//-----------------------------------------------------------------------------
void dGqElement::set(const real* const values)
{
  for (unsigned int i = 0; i <= q; i++)
    this->values[i] = values[i];
}
//-----------------------------------------------------------------------------
void dGqElement::get(real* const values) const
{
  for (unsigned int i = 0; i <= q; i++)
    values[i] = this->values[i];
}
//-----------------------------------------------------------------------------
bool dGqElement::accept(real TOL, real r)
{
  real error = pow(timestep(), static_cast<real>(q+1)) * fabs(r);
  return error <= TOL;
}
//-----------------------------------------------------------------------------
real dGqElement::computeTimeStep(real TOL, real r, real kmax) const
{
  // Compute new time step based on residual

  if ( fabs(r) < DOLFIN_EPS )
    return kmax;

  // FIXME: Missing stability factor and interpolation constant
  return pow(TOL / fabs(r), 1.0 / static_cast<real>(q+1));
}
//-----------------------------------------------------------------------------
real dGqElement::computeDiscreteResidual(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Compute discrete residual
  return (values[q] - u0 - integral(q)) / timestep();
}
//-----------------------------------------------------------------------------
real dGqElement::computeElementResidual(RHS& f)
{
  // Evaluate right-hand side
  feval(f);

  // Compute element residual
  return values[q] - u0 - integral(q);
}
//-----------------------------------------------------------------------------
void dGqElement::feval(RHS& f)
{
  // See comment on cGqElement::feval()

  real k = t1 - t0;
  
  for (unsigned int i = 0; i <= q; i++)
    this->f(i) = f(_index, i, t0 + dG(q).point(i)*k);
}
//-----------------------------------------------------------------------------
real dGqElement::integral(unsigned int i) const
{
  real k = t1 - t0;

  real sum = 0.0;
  for (unsigned int j = 0; j <= q; j++)
    sum += dG(q).weight(i, j) * f(j);

  return k * sum;
}
//-----------------------------------------------------------------------------
