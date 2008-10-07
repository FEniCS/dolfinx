// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2008-10-06

#include <string>
#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/parameters.h>
#include "ODE.h"
#include "Method.h"
#include "MonoAdaptiveFixedPointSolver.h"
#include "MonoAdaptiveNewtonSolver.h"
#include "MonoAdaptiveTimeSlab.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::MonoAdaptiveTimeSlab(ODE& ode)
  : TimeSlab(ode), solver(0), adaptivity(ode, *method), nj(0), dofs(0), 
    fq(0), rmax(0), x(0), u(0), f(0)
{
  // Choose solver
  solver = chooseSolver();

  // Initialize dofs
  dofs = new double[method->nsize()];

  // Compute the number of dofs
  nj = method->nsize() * N;

  // Initialize values of right-hand side
  const uint fsize = method->qsize() * N;
  fq = new double[fsize];
  for (uint j = 0; j < fsize; j++)
    fq[j] = 0.0;

  // Initialize solution
  x = new double[nj];
  real_zero(nj, x);

  // Initialize arrays for u and f
  u = new double[N];
  f = new double[N];

  // Evaluate f at initial data for cG(q)
  if (method->type() == Method::cG)
  {
    ode.f(u0, 0.0, f);
    copy(f, 0, fq, 0, N);
  }
}
//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::~MonoAdaptiveTimeSlab()
{
  delete solver;
  delete [] dofs;
  delete [] fq;
  delete [] x;
  delete [] u;
  delete [] f;
}
//-----------------------------------------------------------------------------
double MonoAdaptiveTimeSlab::build(double a, double b)
{
  //cout << "Mono-adaptive time slab: building between "
  //     << a << " and " << b << endl;

  // Copy initial values to solution
  for (uint n = 0; n < method->nsize(); n++)
    for (uint i = 0; i < N; i++)
      x[n*N + i] = u0[i];

  // Choose time step
  const double k = adaptivity.timestep();
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
  //message("Solving time slab system on [%f, %f].", _a, _b);

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
  const double k = length();
  rmax = 0.0;
  for (uint i = 0; i < N; i++)
  {
    // Prepare data for computation of derivative
    const double x0 = u0[i];
    for (uint n = 0; n < method->nsize(); n++)
      dofs[n] = x[n*N + i];

    // Compute residual
    const double r = fabs(method->residual(x0, dofs, fq[foffset + i], k));
    
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
    copy(x, xoffset, u, 0, N);
    write(N, u);
  }

  // Let user update ODE
  copy(x, xoffset, u, 0, N);
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
void MonoAdaptiveTimeSlab::sample(double t)
{
  // Compute f at end-time
  feval(method->qsize() - 1);
}
//-----------------------------------------------------------------------------
double MonoAdaptiveTimeSlab::usample(uint i, double t)
{
  // Prepare data
  const double x0 = u0[i];
  const double tau = (t - _a) / (_b - _a);  

  // Prepare array of values
  for (uint n = 0; n < method->nsize(); n++)
    dofs[n] = x[n*N + i];

  // Interpolate value
  const double value = method->ueval(x0, dofs, tau);
  
  return value;
}
//-----------------------------------------------------------------------------
double MonoAdaptiveTimeSlab::ksample(uint i, double t)
{
  return length();
}
//-----------------------------------------------------------------------------
double MonoAdaptiveTimeSlab::rsample(uint i, double t)
{
  // Right-hand side at end-point already computed

  // Prepare data for computation of derivative
  const double x0 = u0[i];
  for (uint n = 0; n < method->nsize(); n++)
    dofs[n] = x[n*N + i];
  
  // Compute residual
  const double k = length();
  const uint foffset = (method->qsize() - 1) * N;
  const double r = method->residual(x0, dofs, fq[foffset + i], k);

  return r;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::disp() const
{
  cout << "--- Mono-adaptive time slab ------------------------------" << endl;
  cout << "nj = " << nj << endl;
  cout << "x =";
  for (uint j = 0; j < nj; j++)
    cout << " " << x[j];
  cout << endl;
  cout << "f =";
  for (uint j = 0; j < (method->nsize() * N); j++)
    cout << " " << fq[j];
  cout << endl;
  cout << "----------------------------------------------------------" << endl;
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

    const double t = _a + method->qpoint(m) * (_b - _a);    
    copy(x, (m - 1)*N, u, 0, N);
    ode.f(u, t, f);
    copy(f, 0, fq, m*N, N);
  }
  else
  {
    const double t = _a + method->qpoint(m) * (_b - _a);    
    copy(x, m*N, u, 0, N);
    ode.f(u, t, f);
    copy(f, 0, fq, m*N, N);
  }
}
//-----------------------------------------------------------------------------
TimeSlabSolver* MonoAdaptiveTimeSlab::chooseSolver()
{
  bool implicit = ode.get("ODE implicit");
  std::string solver = ode.get("ODE nonlinear solver");

  if (solver == "fixed-point")
  {
    if (implicit)
      error("Newton solver must be used for implicit ODE.");

    message("Using mono-adaptive fixed-point solver.");
    return new MonoAdaptiveFixedPointSolver(*this);
  }
  else if (solver == "newton")
  {
    if (implicit)
    {
      message("Using mono-adaptive Newton solver for implicit ODE.");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
    else
    {
      message("Using mono-adaptive Newton solver.");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
  }
  else if (solver == "default")
  {
    if (implicit)
    {      
      message("Using mono-adaptive Newton solver (default for implicit ODEs).");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
    else
    {
      message("Using mono-adaptive fixed-point solver (default for c/dG(q)).");
      return new MonoAdaptiveFixedPointSolver(*this);
    }
  }
  else
  {
    error("Uknown solver type: %s.", solver.c_str());
  }

  return 0;
}
//-----------------------------------------------------------------------------
double* MonoAdaptiveTimeSlab::tmp()
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
