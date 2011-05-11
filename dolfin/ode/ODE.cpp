// Copyright (C) 2003-2011 Johan Jansson and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Benjamin Kehlet 2009
//
// First added:  2003-10-21
// Last changed: 2011-03-17

#include <dolfin/log/log.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/constants.h>
#include <dolfin/math/dolfin_math.h>
#include "ODESolver.h"
#include "TimeStepper.h"
#include "ODE.h"
#include "Dual.h"
#include "StabilityAnalysis.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ODE::ODE(uint N, real T)
  : N(N), t(0), T(T), dependencies(N), transpose(N), time_stepper(0), tmp0(N), tmp1(N),
    not_impl_f("Warning: consider implementing mono-adaptive ODE::f() to improve efficiency."),
    not_impl_M("Warning: multiplication with M not implemented, assuming identity."),
    not_impl_J("Warning: consider implementing Jacobian ODE::J() to improve efficiency."),
    not_impl_JT("Warning: consider implementing Jacobian transpose ODE::JT() to improve efficiency")
{
  not_working_in_parallel("ODE solver");

  log(TRACE, "Creating ODE of size %d.", N);
  parameters = default_parameters();

  #ifdef HAS_GMP
  if (!_real_initialized)
      warning("Extended precision not initialized. Use set_precision(uint decimal_prec) before declaring any real variables and instansiating ODE.");
  #endif
}
//-----------------------------------------------------------------------------
ODE::~ODE()
{
  //delete [] tmp0;
  //delete [] tmp1;
  delete time_stepper;
}
//-----------------------------------------------------------------------------
void ODE::f(const Array<real>& u, real t, Array<real>& y)
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
real ODE::f(const Array<real>& u, real t, uint i)
{
  error("Right-hand side for ODE not supplied by user.");
  return 0.0;
}
//-----------------------------------------------------------------------------
void ODE::M(const Array<real>& dx, Array<real>& dy, const Array<real>& u, real t)
{
  // Display a warning, implicit system but M is not implemented
  not_impl_M();

  // Assume M is the identity if not supplied by user: y = x
  real_set(N, dy.data().get(), dx.data().get());
}
//-----------------------------------------------------------------------------
void ODE::J(const Array<real>& dx, Array<real>& dy, const Array<real>& u, real t)
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
    umax = real_max(umax, real_abs(u[i]));
  real h = real_max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  Array<real>& uu = const_cast<Array<real>&>(u);

  // Initialize temporary array if necessary
  //if (!tmp0) tmp0 = new real[N];
  //real_zero(N, tmp0);
  tmp0.zero();

  // Evaluate at u + hx
  real_axpy(N, uu.data().get(), h, dx.data().get());
  f(uu, t, dy);

  // Evaluate at u - hx
  real_axpy(N, uu.data().get(), -2.0*h, dx.data().get());
  f(uu, t, tmp0);

  // Reset u
  real_axpy(N, uu.data().get(), h, dx.data().get());

  // Compute product dy = J dx
  real_sub(N, dy.data().get(), tmp0.data().get());
  real_mult(N, dy.data().get(), 0.5/h);
}
//------------------------------------------------------------------------
void ODE::JT(const Array<real>& dx, Array<real>& dy, const Array<real>& u, real t)
{
  // Display warning
  not_impl_JT();

  // Small change in u
  real umax = 0.0;
  for (unsigned int i = 0; i < N; i++)
    umax = real_max(umax, real_abs(u[i]));
  real h = real_max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * umax);

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  Array<real>& uu = const_cast<Array<real>&>(u);

  // Initialize temporary arrays if necessary
  //if (!tmp0) tmp0 = new real[N];
  //real_zero(N, tmp0);
  tmp0.zero();
  //if (!tmp1) tmp1 = new real[N];
  //real_zero(N, tmp1);
  tmp1.zero();

  // Compute action of transpose of Jacobian
  for (uint i = 0; i < N; ++i)
  {
    uu[i] += h;
    f(uu, t, tmp0);

    uu[i] -= 2*h;
    f(uu, t, tmp1);

    uu[i] += h;

    real_sub(N, tmp0.data().get(), tmp1.data().get());
    real_mult(N, tmp0.data().get(), 0.5/h);

    dy[i] = real_inner(N, tmp0.data().get(), dx.data().get());
  }
}
//------------------------------------------------------------------------
real ODE::dfdu(const Array<real>& u, real t, uint i, uint j)
{
  // Compute Jacobian numerically if dfdu() is not implemented by user

  // FIXME: Maybe we should move this somewhere else?

  // We are not allowed to change u, but we restore it afterwards,
  // so maybe we can cheat a little...
  Array<real>& uu = const_cast<Array<real>&>(u);

  // Save value of u_j
  real uj = uu[j];

  // Small change in u_j
  real h = real_max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * real_abs(uj));

  // Compute F values
  uu[j] -= 0.5 * h;
  real f1 = f(uu, t, i);

  uu[j] = uj + 0.5*h;
  real f2 = f(uu, t, i);

  // Reset value of uj
  uu[j] = uj;

  // Compute derivative
  if (real_abs(f1 - f2) < real_epsilon() * real_max(real_abs(f1), real_abs(f2)))
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
bool ODE::update(const Array<real>& u, real t, bool end)
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
real ODE::time() const
{
  return t;
}
//-----------------------------------------------------------------------------
real ODE::time(real t) const
{
  return t;
}
//-----------------------------------------------------------------------------
real ODE::endtime() const
{
  return T;
}
//-----------------------------------------------------------------------------
void ODE::sparse()
{
  dependencies.detect(*this);
}
//-----------------------------------------------------------------------------
void ODE::solve()
{
  assert(!time_stepper);

  // Solve ODE on entire time interval
  ODESolver ode_solver(*this);
  ode_solver.solve();
}
//-----------------------------------------------------------------------------
void ODE::solve(real t0, real t1)
{
  // Check time interval
  if (t0 < 0.0 - real_epsilon() || t1 > endtime() + real_epsilon())
  {
    error("Illegal time interval [%g, %g] for ODE system, not contained in [%g, %g].",
          to_double(t0), to_double(t1), to_double(0.0), to_double(endtime()));
  }

  // Create time stepper if not created before
  if (!time_stepper)
    time_stepper = new TimeStepper(*this);

  // Solve ODE on given time interval
  time_stepper->solve(t0, t1);
}
//-----------------------------------------------------------------------------
void ODE::solve(ODESolution& u)
{
  assert(!time_stepper);

  // Solve ODE on entire time interval
  ODESolver ode_solver(*this);
  ode_solver.solve(u);
}
//-----------------------------------------------------------------------------
void ODE::solve(ODESolution& u, real t0, real t1)
{
  // Check time interval
  if (t0 < 0.0 - real_epsilon() || t1 > endtime() + real_epsilon())
  {
    error("Illegal time interval [%g, %g] for ODE system, not contained in [%g, %g].",
          to_double(t0), to_double(t1), to_double(0.0), to_double(endtime()));
  }

  // Create time stepper if not created before
  if (!time_stepper)
    time_stepper = new TimeStepper(*this, u);

  // Solve ODE on given time interval
  time_stepper->solve(t0, t1);
}
//-----------------------------------------------------------------------------
void ODE::solve_dual(ODESolution& u) {
  begin("Solving dual problem");

  // Create dual problem
  Dual dual(*this, u);

  // Solve dual problem
  dual.solve();

  end();
}
//-----------------------------------------------------------------------------
void ODE::solve_dual(ODESolution& u, ODESolution& z) {
  begin("Solving dual problem");

  // Create dual problem
  Dual dual(*this, u);

  // Solve dual problem
  dual.solve(z);

  end();
}
//-----------------------------------------------------------------------------
void ODE::analyze_stability( uint q, ODESolution& u) {
  StabilityAnalysis S(*this, u);
  S.analyze_integral(q);
}
//-----------------------------------------------------------------------------
void ODE::analyze_stability_discretization(ODESolution& u) {
  StabilityAnalysis S(*this, u);
  S.analyze_integral(parameters["order"]);
}
//-----------------------------------------------------------------------------
void ODE::analyze_stability_computation(ODESolution& u) {
  StabilityAnalysis S(*this, u);
  S.analyze_integral(0);
}
//-----------------------------------------------------------------------------
void ODE::analyze_stability_initial(ODESolution& u) {
  StabilityAnalysis S(*this, u);
  S.analyze_endpoint();
}
//-----------------------------------------------------------------------------
void ODE::set_state(const Array<real>& u)
{
  // Create time stepper if not created before
  if (!time_stepper)
    time_stepper = new TimeStepper(*this);

  // Set state
  time_stepper->set_state(u.data().get());
}
//-----------------------------------------------------------------------------
void ODE::get_state(Array<real>& u)
{
  // Create time stepper if not created before
  if (!time_stepper)
    time_stepper = new TimeStepper(*this);

  // Get state
  time_stepper->get_state(u.data().get());
}
//-----------------------------------------------------------------------------
