// Copyright (C) 2003-2008 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2008-10-18

#include <dolfin/common/constants.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/parameter/parameters.h>
#include "ODESolver.h"
#include "ODE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ODE::ODE(uint N, real T)
  : N(N), T(T), dependencies(N), transpose(N), tmp0(0), tmp1(0),
    not_impl_f("Warning: consider implementing mono-adaptive ODE::f() to improve efficiency."),
    not_impl_M("Warning: multiplication with M not implemented, assuming identity."),
    not_impl_J("Warning: consider implementing Jacobian ODE::J() to improve efficiency."),
    not_impl_JT("Warning: consider implementing Jacobian transpose ODE::JT() to improve efficiency")
{
  message("Creating ODE of size %d.", N);

 #ifdef HAS_GMP
  char msg[100];
  gmp_sprintf(msg, "ODE::epsilon=%Fe", ODE::epsilon().get_mpf_t());
  message(msg);
  message("GMP: Using %d bits pr number", mpf_get_default_prec());
 #endif
}
//-----------------------------------------------------------------------------
ODE::~ODE()
{
  delete [] tmp0;
  delete [] tmp1;
}
//-----------------------------------------------------------------------------
void ODE::f(const real* u, real t, real* y)
{
  // If a user of the mono-adaptive solver does not supply this function,
  // then call f_i() for each component.
  
  // Display a warning, more efficiently if implemented
  not_impl_f();

  // Call f for each component
  for (uint i = 0; i < N; i++)
    y[i] = this->f(u, t, i);
}
//-----------------------------------------------------------------------------
real ODE::f(const real* u, real t, uint i)
{
  error("Right-hand side for ODE not supplied by user.");
  return 0.0;
}
//-----------------------------------------------------------------------------
void ODE::M(const real* x, real* y, const real* u, real t)
{
  // Display a warning, implicit system but M is not implemented
  not_impl_M();

  // Assume M is the identity if not supplied by user: y = x
  real_set(N, y, x);
}
//-----------------------------------------------------------------------------
void ODE::J(const real* x, real* y, const real* u, real t)
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
    umax = max(umax, abs(u[i]));
  real h = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  real* uu = const_cast<real*>(u);

  // Initialize temporary array if necessary
  if (!tmp0) tmp0 = new real[N];
  real_zero(N, tmp0);

  // Evaluate at u + hx
  real_axpy(N, uu, h, x);
  f(uu, t, y);

  // Evaluate at u - hx
  real_axpy(N, uu, -2.0*h, x);
  f(uu, t, tmp0);

  // Reset u
  real_axpy(N, uu, h, x);

  // Compute product y = Jx
  real_sub(N, y, tmp0);
  real_mult(N, y, 0.5/h);
}
//------------------------------------------------------------------------
void ODE::JT(const real* x, real* y, const real* u, real t)
{
  // Display warning
  not_impl_JT();

  // Small change in u
  real umax = 0.0;
  for (unsigned int i = 0; i < N; i++)
    umax = max(umax, abs(u[i]));
  real h = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  real* uu = const_cast<real*>(u);

  // Initialize temporary arrays if necessary
  if (!tmp0) tmp0 = new real[N];
  real_zero(N, tmp0);
  if (!tmp1) tmp1 = new real[N];
  real_zero(N, tmp1);

  // Compute action of transpose of Jacobian
  for (uint i = 0; i < N; ++i)
  {
    uu[i] += h;
    f(uu, t, tmp0);
    
    uu[i] -= 2*h;
    f(uu, t, tmp1);

    uu[i] += h;

    real_sub(N, tmp0, tmp1);
    real_mult(N, tmp0, 0.5/h);

    y[i] = real_inner(N, tmp0, x);
  }
}
//------------------------------------------------------------------------
real ODE::dfdu(const real* u, real t, uint i, uint j)
{
  // Compute Jacobian numerically if dfdu() is not implemented by user
  
  // FIXME: Maybe we should move this somewhere else?

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  real* uu = const_cast<real*>(u);

  // Save value of u_j
  real uj = uu[j];
  
  // Small change in u_j
  real h = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * abs(uj));
  
  // Compute F values
  uu[j] -= 0.5 * h;
  real f1 = f(uu, t, i);
  
  uu[j] = uj + 0.5*h;
  real f2 = f(uu, t, i);
         
  // Reset value of uj
  uu[j] = uj;

  // Compute derivative
  if ( abs(f1 - f2) < ODE::epsilon() * max(abs(f1), abs(f2)) )
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
bool ODE::update(const real* u, real t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------
void ODE::save(Sample& sample)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real ODE::time(real t) const
{
  return t;
}
//-----------------------------------------------------------------------------
void ODE::sparse()
{
  dependencies.detect(*this);
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

#ifdef HAS_GMP
//initialize to 10e-30
real ODE::_epsilon = real("0.000000000000000000000000000001");
#else
real ODE::_epsilon = DOLFIN_EPS;
#endif
