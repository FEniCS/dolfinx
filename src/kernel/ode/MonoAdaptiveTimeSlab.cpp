// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-28
// Last changed: 2006-07-04

#ifdef HAVE_PETSC_H

#include <string>
#include <dolfin/dolfin_log.h>
#include <dolfin/ParameterSystem.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MonoAdaptiveFixedPointSolver.h>
#include <dolfin/MonoAdaptiveNewtonSolver.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::MonoAdaptiveTimeSlab(ODE& ode)
  : TimeSlab(ode), solver(0), adaptivity(ode, *method), nj(0), dofs(0), 
    fq(0), rmax(0)
{
  // Choose solver
  solver = chooseSolver();

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
  x.init(nj);

  // Evaluate f at initial data for cG(q)
  if ( method->type() == Method::cG )
  {
    ode.f(u0, 0.0, f);
    copy(f, 0, fq, 0, N);
  }
}
//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::~MonoAdaptiveTimeSlab()
{
  if ( solver ) delete solver;
  if ( dofs ) delete [] dofs;
  if ( fq ) delete [] fq;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::build(real a, real b)
{
  //cout << "Mono-adaptive time slab: building between "
  //     << a << " and " << b << endl;

  // Copy initial values to solution
  real* xx = x.array();
  for (uint n = 0; n < method->nsize(); n++)
  {
    for (uint i = 0; i < N; i++)
    {
      xx[n*N + i] = u0[i];
    }
  }
  x.restore(xx);

  // Choose time step
  const real k = adaptivity.timestep();
  if ( k < adaptivity.threshold() * (b - a) )
    b = a + k;

  // Save start and end time
  _a = a;
  _b = b;

  //cout << "Mono-adaptive time slab: finished building between "
  //     << a << " and " << b << ": K = " << b - a << endl;

  // Update at t = 0.0
  if ( a < DOLFIN_EPS )
    ode.update(u0, a, false);

  return b;
}
//-----------------------------------------------------------------------------
bool MonoAdaptiveTimeSlab::solve()
{
  //dolfin_info("Solving time slab system on [%f, %f].", _a, _b);

  return solver->solve();
}
//-----------------------------------------------------------------------------
bool MonoAdaptiveTimeSlab::check(bool first)
{
  // Get array
  real* xx = x.array();

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
      dofs[n] = xx[n*N + i];

    // Compute residual
    const real r = fabs(method->residual(x0, dofs, fq[foffset + i], k));
    
    // Compute maximum
    if ( r > rmax )
      rmax = r;
  }

  // Restore array
  x.restore(xx);

  // Compute new time step
  adaptivity.update(length(), rmax, *method, _b, first);

  // Check if current solution can be accepted
  return adaptivity.accept();
}
//-----------------------------------------------------------------------------
bool MonoAdaptiveTimeSlab::shift()
{
  // Compute offsets
  const uint xoffset = (method->nsize() - 1) * N;
  const uint foffset = (method->qsize() - 1) * N;

  // Get array
  real* xx = x.array();
  
  // Check if we reached the end time
  const bool end = (_b + DOLFIN_EPS) > ode.T;

  // Write solution at final time if we should
  if ( save_final && end )
    write(xx + xoffset);

  // Let user update ODE
  if ( !ode.update(xx + xoffset, _b, end) )
  {
    x.restore(xx);
    return false;
  }

  // Set initial value to end-time value
  for (uint i = 0; i < N; i++)
    u0[i] = xx[xoffset + i];

  // Set f at first quadrature point to f at end-time for cG(q)
  if ( method->type() == Method::cG )
  {
    for (uint i = 0; i < N; i++)
      fq[i] = fq[foffset + i];
  }

  // Restore array
  x.restore(xx);

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
  real* xx = x.array();

  // Prepare array of values
  for (uint n = 0; n < method->nsize(); n++)
    dofs[n] = xx[n*N + i];

  // Interpolate value
  const real value = method->ueval(x0, dofs, tau);
  
  // Restore array
  x.restore(xx);

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
  real* xx = x.array();
  const real x0 = u0[i];
  for (uint n = 0; n < method->nsize(); n++)
    dofs[n] = xx[n*N + i];
  x.restore(xx);
  
  // Compute residual
  const real k = length();
  const uint foffset = (method->qsize() - 1) * N;
  const real r = method->residual(x0, dofs, fq[foffset + i], k);

  return r;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::disp() const
{
  cout << "--- Mono-adaptive time slab ------------------------------" << endl;
  cout << "nj = " << nj << endl;
  cout << "x =";
  const real* xx = x.array();
  for (uint j = 0; j < nj; j++)
    cout << " " << xx[j];
  cout << endl;
  x.restore(xx);
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
  if ( method->type() == Method::cG )
  {
    // Special case: m = 0
    if ( m == 0 )
    {
      // We don't need to evaluate f at t = a since we evaluated
      // f at t = b for the previous time slab
      return;
    }

    const real t = _a + method->qpoint(m) * (_b - _a);    
    real* xx = x.array();
    copy(xx, (m - 1)*N, u, 0, N);
    ode.f(u, t, f);
    copy(f, 0, fq, m*N, N);
    x.restore(xx);
  }
  else
  {
    const real t = _a + method->qpoint(m) * (_b - _a);    
    real* xx = x.array();
    copy(xx, m*N, u, 0, N);
    ode.f(u, t, f);
    copy(f, 0, fq, m*N, N);
    x.restore(xx);
  }
}
//-----------------------------------------------------------------------------
TimeSlabSolver* MonoAdaptiveTimeSlab::chooseSolver()
{
  bool implicit = get("ODE implicit");
  std::string solver = get("ODE nonlinear solver");

  if ( solver == "fixed-point" )
  {
    if ( implicit )
      dolfin_error("Newton solver must be used for implicit ODE.");

    dolfin_info("Using mono-adaptive fixed-point solver.");
    return new MonoAdaptiveFixedPointSolver(*this);
  }
  else if ( solver == "newton" )
  {
    if ( implicit )
    {
      dolfin_info("Using mono-adaptive Newton solver for implicit ODE.");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
    else
    {
      dolfin_info("Using mono-adaptive Newton solver.");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
  }
  else if ( solver == "default" )
  {
    if ( implicit )
    {      
      dolfin_info("Using mono-adaptive Newton solver (default for implicit ODEs).");
      return new MonoAdaptiveNewtonSolver(*this, implicit);
    }
    else
    {
      dolfin_info("Using mono-adaptive fixed-point solver (default for c/dG(q)).");
      return new MonoAdaptiveFixedPointSolver(*this);
    }
  }
  else
  {
    dolfin_error1("Uknown solver type: %s.", solver.c_str());
  }

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

  if ( method->type() == Method::cG )
    return fq + N;
  else
    return fq;
}
//-----------------------------------------------------------------------------

#endif
