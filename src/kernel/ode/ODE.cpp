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
ODE::ODE(uint N) : N(N), T(1.0), sparsity(N), dependencies(N), transpose(N),
		   default_timestep(dolfin_get("initial time step")),
		   default_order(dolfin_get("order")),
		   not_impl_f("Warning: consider implementing mono-adaptive ODE::f() to improve efficiency."),
		   not_impl_M("Warning: multiplication with M not implemented, assuming identity."),
		   not_impl_J("Warning: consider implementing ODE::J() to improve efficiency.")
{
  dolfin_info("Creating ODE of size %d.", N);

  // Choose method
  string method = dolfin_get("method");
  if ( method == "cg" )
    default_method = Element::cg;
  else
    default_method = Element::dg;
}
//-----------------------------------------------------------------------------
ODE::~ODE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real ODE::f(const real u[], real t, uint i)
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
void ODE::f(const real u[], real t, real y[])
{
  // If a user of the mono-adaptive solver does not supply this function,
  // then call f() for each component.

  // Display a warning, more efficiently if implemented
  not_impl_f();

  // Call f for each component
  for (uint i = 0; i < N; i++)
    y[i] = this->f(u, t, i);
}
//-----------------------------------------------------------------------------
void ODE::M(const real x[], real y[], const real u[], real t)
{
  // Assume M is the identity if not supplied by user: y = x
  
  // Display a warning, implicit system but M is not implemented
  not_impl_M();

  // Set y = x
  for (uint i = 0; i < N; i++)
    y[i] = x[i];
}
//-----------------------------------------------------------------------------
void ODE::J(const real x[], real y[], const real u[], real t)
{
  // If a user does not supply J, then compute it by the approximation
  //
  //     Jx = ( f(z + hx) - f(z - hx) ) / 2h

  // Display a warning, more efficiently if implemented
  not_impl_J();

  dolfin_error("Not implemented yet...");

  /* Old version, should not be used
  
  // Compute product
  for (uint i = 0; i < N; i++)
  {
    real sum = 0.0;
    const NewArray<uint>& deps = dependencies[i];
    for (uint pos = 0; pos < deps.size(); pos++)
    {
      const uint j = deps[pos];
      sum += dfdu(u, t, i, j) * x[j];
    }
    y[i] = sum;
  }

  */
}
//-----------------------------------------------------------------------------
real ODE::dfdu(const real u[], real t, uint i, uint j)
{
  // Compute Jacobian numerically if dfdu() is not implemented by user
  
  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  real* uu = const_cast<real*>(u);

  // Save value of u_j
  real uj = uu[j];
  
  // Small change in u_j
  real dU = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * abs(uj));
  
  // Compute F values
  uu[j] -= 0.5 * dU;
  real f1 = f(uu, t, i);
  
  uu[j] = uj + 0.5*dU;
  real f2 = f(uu, t, i);
         
  // Reset value of uj
  uu[j] = uj;

  // Compute derivative
  if ( abs(f1 - f2) < DOLFIN_EPS * max(abs(f1), abs(f2)) )
    return 0.0;

  return (f2 - f1) / dU;
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
real ODE::timestep()
{
  return default_timestep;
}
//-----------------------------------------------------------------------------
real ODE::timestep(uint i)
{
  return timestep();
}
//-----------------------------------------------------------------------------
bool ODE::update(const real u[], real t, bool end)
{
  return true;
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
