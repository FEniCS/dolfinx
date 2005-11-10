// Copyright (C) 2003-2005 Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2005-10-24

#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/ODESolver.h>
#include <dolfin/ODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ODE::ODE(uint N, real T)
  : N(N), T(T), dependencies(N), transpose(N), tmp(0),
    not_impl_f("Warning: consider implementing mono-adaptive ODE::f() to improve efficiency."),
    not_impl_M("Warning: multiplication with M not implemented, assuming identity."),
    not_impl_J("Warning: consider implementing ODE::J() to improve efficiency.")
{
  dolfin_info("Creating ODE of size %d.", N);
}
//-----------------------------------------------------------------------------
ODE::~ODE()
{
  if ( tmp ) delete [] tmp;
}
//-----------------------------------------------------------------------------
real ODE::f(const real u[], real t, uint i)
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
  //     Jx = ( f(u + hx) - f(u - hx) ) / 2h

  // Display a warning, more efficiently if implemented
  not_impl_J();

  // Small change in u
  real umax = 0.0;
  for (unsigned int i = 0; i < N; i++)
    umax = std::max(umax, std::abs(u[i]));
  real h = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  real* uu = const_cast<real*>(u);

  // Allocate temporary array if not already allocated
  if ( !tmp )
    tmp = new real[N];
  
  // Evaluate at u + hx
  for (unsigned int i = 0; i < N; i++)
    uu[i] += h*x[i];
  f(uu, t, y);

  // Evaluate at u - hx
  for (unsigned int i = 0; i < N; i++)
    uu[i] -= 2.0*h*x[i];
  f(uu, t, tmp);

  // Reset u
  for (unsigned int i = 0; i < N; i++)
    uu[i] += h*x[i];

  // Compute product y = Jx
  for (unsigned int i = 0; i < N; i++)
    y[i] = (y[i] - tmp[i]) / (2.0*h);
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
  real h = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * std::abs(uj));
  
  // Compute F values
  uu[j] -= 0.5 * h;
  real f1 = f(uu, t, i);
  
  uu[j] = uj + 0.5*h;
  real f2 = f(uu, t, i);
         
  // Reset value of uj
  uu[j] = uj;

  // Compute derivative
  if ( std::abs(f1 - f2) < DOLFIN_EPS * std::max(std::abs(f1), std::abs(f2)) )
    return 0.0;

  return (f2 - f1) / h;
}
//-----------------------------------------------------------------------------
real ODE::timestep(real t) const
{
  dolfin_error("Time step should be fixed but has not been specified.");
  return 0.0;
}
//-----------------------------------------------------------------------------
real ODE::timestep(real t, uint i) const
{
  dolfin_error("Time step should be fixed but has not been specified.");
  return 0.0;
}
//-----------------------------------------------------------------------------
bool ODE::update(const real u[], real t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------
void ODE::save(Sample& sample)
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
void ODE::sparse()
{
  dependencies.detect(*this);
}
//-----------------------------------------------------------------------------
void ODE::sparse(const Matrix& A)
{
  dependencies.set(A);
}
//-----------------------------------------------------------------------------
