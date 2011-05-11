// Copyright (C) 2005-2008 Anders Logg
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
// First added:  2005-01-28
// Last changed: 2009-09-08

#include <string>
#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include "ODE.h"
#include "ODESolution.h"
#include "Method.h"
#include "MonoAdaptiveFixedPointSolver.h"
#include "MonoAdaptiveNewtonSolver.h"
#include "MonoAdaptiveTimeSlab.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::MonoAdaptiveTimeSlab(ODE& ode)
  : TimeSlab(ode), solver(0), adaptivity(ode, *method), nj(0), dofs(0),
    fq(0), rmax(0), x(0), u(N), f(N)
{
  // Choose solver
  solver = choose_solver();

  // Initialize dofs
  dofs = new real[method->nsize()];

  // Compute the number of dofs
  nj = method->nsize() * N;

  // Initialize values of right-hand side
  const uint fsize = method->qsize() * N;
  fq = new real[fsize];
  for (uint j = 0; j < fsize; j++)
    fq[j] = 0.0;

  // Initialize solution
  x = new real[nj];
  real_zero(nj, x);

  // Initialize arrays for u and f
  //u = new real[N];
  //f = new real[N];

  // Evaluate f at initial data for cG(q)
  if (method->type() == Method::cG)
  {
    ode.f(u0, 0.0, f);
    copy(f, fq, 0);
    //copy(f, 0, fq, 0, N);
  }
}
//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::~MonoAdaptiveTimeSlab()
{
  delete solver;
  delete [] dofs;
  delete [] fq;
  delete [] x;
  //delete [] u;
  //delete [] f;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::build(real a, real b)
{
  //cout << "Mono-adaptive time slab: building between "
  //     << a << " and " << b << endl;

  // Copy initial values to solution
  for (uint n = 0; n < method->nsize(); n++)
    for (uint i = 0; i < N; i++)
      x[n*N + i] = u0[i];

  // Choose time step
  const real k = adaptivity.timestep();
  if (k < adaptivity.threshold() * (b - a))
    b = a + k;

  // Save start and end time
  _a = a;
  _b = b;

  //cout << "Mono-adaptive time slab: finished building between "
  //     << a << " and " << b << ": K = " << b - a << endl;

  // Update at t = 0.0
  if (a < DOLFIN_EPS)
    ode.update(u0, a, false);

  return b;
}
//-----------------------------------------------------------------------------
bool MonoAdaptiveTimeSlab::solve()
{
  //info("Solving time slab system on [%f, %f].", _a, _b);

  return solver->solve();
}
//-----------------------------------------------------------------------------
bool MonoAdaptiveTimeSlab::check(bool first)
{
  // Compute offset for f
  const uint foffset = (method->qsize() - 1) * N;

  // Compute f at end-time
  feval(method->qsize() - 1);

  // Compute maximum norm of residual at end-time
  const real k = length();
  rmax = 0.0;
  for (uint i = 0; i < N; i++)
  {
    // Prepare data for computation of derivative
    const real x0 = u0[i];
    for (uint n = 0; n < method->nsize(); n++)
      dofs[n] = x[n*N + i];

    // Compute residual
    const real r = real_abs(method->residual(x0, dofs, fq[foffset + i], k));

    // Compute maximum
    if (r > rmax)
      rmax = r;
  }

  // Compute new time step
  adaptivity.update(length(), rmax, *method, _b, first);

  // Check if current solution can be accepted
  return adaptivity.accept();
}
//-----------------------------------------------------------------------------
bool MonoAdaptiveTimeSlab::shift(bool end)
{
  // Compute offsets
  const uint xoffset = (method->nsize() - 1) * N;
  const uint foffset = (method->qsize() - 1) * N;

  // Write solution at final time if we should
  if (save_final && end)
  {
    copy(x, xoffset, u);
    write(u);
  }

  // Let user update ODE
  copy(x, xoffset, u);
  if (!ode.update(u, _b, end))
    return false;

  // Set initial value to end-time value
  for (uint i = 0; i < N; i++)
    u0[i] = x[xoffset + i];

  // Set f at first quadrature point to f at end-time for cG(q)
  if (method->type() == Method::cG)
  {
    for (uint i = 0; i < N; i++)
      fq[i] = fq[foffset + i];
  }

  return true;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::sample(real t)
{
  // Compute f at end-time
  feval(method->qsize() - 1);
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::usample(uint i, real t)
{
  // Prepare data
  const real x0 = u0[i];
  const real tau = (t - _a) / (_b - _a);

  // Prepare array of values
  for (uint n = 0; n < method->nsize(); n++)
    dofs[n] = x[n*N + i];

  // Interpolate value
  const real value = method->ueval(x0, dofs, tau);

  return value;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::ksample(uint i, real t)
{
  return length();
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::rsample(uint i, real t)
{
  // Right-hand side at end-point already computed

  // Prepare data for computation of derivative
  const real x0 = u0[i];
  for (uint n = 0; n < method->nsize(); n++)
    dofs[n] = x[n*N + i];

  // Compute residual
  const real k = length();
  const uint foffset = (method->qsize() - 1) * N;
  const real r = method->residual(x0, dofs, fq[foffset + i], k);

  return r;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::save_solution(ODESolution& u)
{
  //printf("MonoAdaptiveTimeSlab::save_solution\n");
  // Prepare array of values
  std::vector<real> data(N*u.nsize());

  for (uint i = 0; i < N; ++i)
  {
    for (uint n = 0; n < method->nsize(); n++)
      dofs[n] = x[n*N + i];

    method->get_nodal_values(u0[i], dofs, &data[i*u.nsize()]);
  }

  u.add_timeslab(starttime(), endtime(), &data[0]);
}
//-----------------------------------------------------------------------------
std::string MonoAdaptiveTimeSlab::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << "  nj = " << nj << std::endl;;
    s << "  x =";
    for (uint j = 0; j < nj; j++)
      s << " " << x[j];
    s << std::endl;;
    s << "  f =";
    for (uint j = 0; j < (method->nsize() * N); j++)
      s << " " << fq[j];
    s << std::endl;;
  }
  else
    s << "<MonoAdaptiveTimeSlab with " << N << " components>";

  return s.str();
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::feval(uint m)
{
  // Evaluation depends on the choice of method
  if (method->type() == Method::cG)
  {
    // Special case: m = 0
    if (m == 0)
    {
      // We don't need to evaluate f at t = a since we evaluated
      // f at t = b for the previous time slab
      return;
    }

    const real t = _a + method->qpoint(m)*(_b - _a);
    copy(x, (m - 1)*N, u);
    ode.f(u, t, f);
    copy(f, fq, m*N);
  }
  else
  {
    const real t = _a + method->qpoint(m) * (_b - _a);
    //copy(x, m*N, u, 0, N);
    //ode.f(&x[m*N], t, &fq[m*N]);
    Array<real> u(N, &x[m*N]);
    Array<real> y(N, &fq[m*N]);
    ode.f(u, t, y);
    //copy(f, 0, fq, m*N, N);
  }
}
//-----------------------------------------------------------------------------
TimeSlabSolver* MonoAdaptiveTimeSlab::choose_solver()
{
  bool implicit = ode.parameters["implicit"];
  std::string solver = ode.parameters["nonlinear_solver"];

  if (solver == "fixed-point")
  {
    if (implicit)
      error("Newton solver must be used for implicit ODE.");

    info("Using mono-adaptive fixed-point solver.");
    return new MonoAdaptiveFixedPointSolver(*this);
  }
  else if (solver == "newton")
  {
    if (implicit)
    {
      info("Using mono-adaptive Newton solver for implicit ODE.");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
    else
    {
      info("Using mono-adaptive Newton solver.");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
  }
  else if (solver == "default")
  {
    if (implicit)
    {
      info("Using mono-adaptive Newton solver (default for implicit ODEs).");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
    else
    {
      info("Using mono-adaptive fixed-point solver (default for c/dG(q)).");
      return new MonoAdaptiveFixedPointSolver(*this);
    }
  }
  else
    error("Uknown solver type: %s.", solver.c_str());

  return 0;
}
//-----------------------------------------------------------------------------
real* MonoAdaptiveTimeSlab::tmp()
{
  // This function provides access to an array that can be used for
  // temporary data storage by the Newton solver. We can reuse the
  // parts of f that are recomputed in each iteration. Note that this
  // needs to be done differently for cG and dG, since cG does not
  // recompute the right-hand side at the first quadrature point.

  if (method->type() == Method::cG)
    return fq + N;
  else
    return fq;
}
//-----------------------------------------------------------------------------
