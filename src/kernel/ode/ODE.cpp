// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Function.h>
#include <dolfin/Vector.h>
#include <dolfin/ODESolver.h>
#include <dolfin/ODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ODE::ODE(uint N) : N(N), T(1.0), sparsity(N), dependencies(N), transpose(N)
{
  // Choose method
  string method = dolfin_get("method");
  if ( method == "cg" )
    default_method = Element::cg;
  else
    default_method = Element::dg;
  
  // Choose order
  default_order = dolfin_get("order");

  // Choose time step
  default_timestep = dolfin_get("initial time step");
}
//-----------------------------------------------------------------------------
ODE::~ODE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real ODE::f(real u[], real t, uint i)
{
  dolfin_error("Not implemented");

  return 0.0;
}
//-----------------------------------------------------------------------------
real ODE::f(const Vector& u, real t, uint i)
{
  dolfin_error("Not implemented");
  
  return 0.0;
}
//-----------------------------------------------------------------------------
real ODE::dfdu(const Vector& u, real t, uint i, uint j)
{
  // Compute Jacobian numerically if dfdu() is not implemented by user

  // We cast away the constness of u, since we need to change the Vector u
  // in one position temporarily. Once the derivative has been computed, we
  // restore u to its original value, keeping it const. 
  Vector& v = const_cast<Vector&>(u);

  // Save value of u_j
  real uj = v(j);

  // Small change in u_j
  real dU = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * abs(uj));
  
  // Compute F values
  v(j) -= 0.5 * dU;
  real f1 = f(v, t, i);
  
  v(j) = uj + 0.5*dU;
  real f2 = f(v, t, i);
         
  // Compute derivative
  if ( abs(f1 - f2) < DOLFIN_EPS * max(abs(f1), abs(f2)) )
    return 0.0;

  v(j) = uj;

  return (f2 - f1) / dU;
}
//-----------------------------------------------------------------------------
Element::Type ODE::method(uint i)
{
  return default_method;
}
//-----------------------------------------------------------------------------
dolfin::uint ODE::order(uint i)
{
  return default_order;
}
//-----------------------------------------------------------------------------
real ODE::timestep(uint i)
{
  return default_timestep;
}
//-----------------------------------------------------------------------------
void ODE::update(RHS& f, Function& u, real t)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ODE::update(Solution& u, Adaptivity& adaptivity, real t)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ODE::save(Sample& sample)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ODE::save(NewSample& sample)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint ODE::size() const
{
  return N;  
}
//-----------------------------------------------------------------------------
real ODE::endtime() const
{
  return T;
}
//-----------------------------------------------------------------------------
void ODE::solve()
{
  ODESolver::solve(*this);
}
//-----------------------------------------------------------------------------
void ODE::solve(Function& u)
{
  ODESolver::solve(*this, u);
}
//-----------------------------------------------------------------------------
void ODE::solve(Function& u, Function& phi)
{
  ODESolver::solve(*this, u, phi);
}
//-----------------------------------------------------------------------------
void ODE::sparse()
{
  // FIXME: Change to new version
  sparsity.detect(*this);
}
//-----------------------------------------------------------------------------
void ODE::sparse(const Matrix& A)
{
  // FIXME: Change to new version
  sparsity.set(A);
}
//-----------------------------------------------------------------------------
