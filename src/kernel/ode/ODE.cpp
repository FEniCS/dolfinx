// Copyright (C) 2003-2006 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2006-07-05

#include <dolfin/dolfin_math.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/ODESolver.h>
#include <dolfin/ODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ODE::ODE(uint N, real T)
  : N(N), T(T), dependencies(N), transpose(N), tmp(0),
    not_impl_f("Warning: consider implementing mono-adaptive ODE::f() to improve efficiency."),
    not_impl_M("Warning: multiplication with M not implemented, assuming identity."),
    not_impl_J("Warning: consider implementing Jacobian ODE::J() to improve efficiency.")
{
  dolfin_info("Creating ODE of size %d.", N);
}
//-----------------------------------------------------------------------------
ODE::~ODE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ODE::f(const uBlasVector& u, real t, uBlasVector& y)
{
  // If a user of the mono-adaptive solver does not supply this function,
  // then call f_i() for each component.

  // Display a warning, more efficiently if implemented
  not_impl_f();

  // Call f for each component
  for (uint i = 0; i < N; i++)
    y(i) = this->f(u, t, i);
}
//-----------------------------------------------------------------------------
real ODE::f(const uBlasVector& u, real t, uint i)
{
  dolfin_error("Right-hand side for ODE not supplied by user.");
  return 0.0;
}
//-----------------------------------------------------------------------------
void ODE::M(const uBlasVector& x, uBlasVector& y, const uBlasVector& u, real t)
{
  // Display a warning, implicit system but M is not implemented
  not_impl_M();

  // Assume M is the identity if not supplied by user: y = x
  y = x;
}
//-----------------------------------------------------------------------------
void ODE::J(const uBlasVector& x, uBlasVector& y, const uBlasVector& u, real t)
{
  // If a user does not supply J, then compute it by the approximation
  //
  //     Jx = ( f(u + hx) - f(u - hx) ) / 2h

  // FIXME: Maybe we should move this somewhere else?

  // Display a warning, more efficiently if implemented
  not_impl_J();

  // Small change in u
  real umax = 0.0;
  for (unsigned int i = 0; i < N; i++)
    umax = std::max(umax, std::abs(u(i)));
  real h = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  uBlasVector& uu = const_cast<uBlasVector&>(u);

  // Initialize temporary array if necessary
  if ( tmp.size() != N )
    tmp.init(N);
  
  // Evaluate at u + hx
  noalias(uu) += h*x;
  f(uu, t, y);

  // Evaluate at u - hx
  noalias(uu) -= 2.0*h*x;
  f(uu, t, tmp);

  // Reset u
  noalias(uu) += h*x;

  // Compute product y = Jx
  noalias(y) -= tmp;
  y *= 0.5/h;
}
//-----------------------------------------------------------------------------
real ODE::dfdu(const uBlasVector& u, real t, uint i, uint j)
{
  // Compute Jacobian numerically if dfdu() is not implemented by user
  
  // FIXME: Maybe we should move this somewhere else?

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  uBlasVector& uu = const_cast<uBlasVector&>(u);

  // Save value of u_j
  real uj = uu(j);
  
  // Small change in u_j
  real h = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * std::abs(uj));
  
  // Compute F values
  uu(j) -= 0.5 * h;
  real f1 = f(uu, t, i);
  
  uu(j) = uj + 0.5*h;
  real f2 = f(uu, t, i);
         
  // Reset value of uj
  uu(j) = uj;

  // Compute derivative
  if ( std::abs(f1 - f2) < DOLFIN_EPS * std::max(std::abs(f1), std::abs(f2)) )
    return 0.0;

  return (f2 - f1) / h;
}
//-----------------------------------------------------------------------------
real ODE::timestep(real t, real k0) const
{
  // Keep old time step by default when "fixed time step" is set
  // and user has not overloaded this function
  return k0;
}
//-----------------------------------------------------------------------------
real ODE::timestep(real t, uint i, real k0) const
{
  // Keep old time step by default when "fixed time step" is set
  // and user has not overloaded this function
  return k0;
}
//-----------------------------------------------------------------------------
bool ODE::update(const uBlasVector& u, real t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------
void ODE::save(Sample& sample)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ODE::sparse()
{
  dependencies.detect(*this);
}
//-----------------------------------------------------------------------------
void ODE::sparse(const uBlasSparseMatrix& A)
{
  dependencies.set(A);
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
