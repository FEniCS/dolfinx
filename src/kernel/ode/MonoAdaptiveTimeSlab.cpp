// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>
#include <dolfin/dolfin_settings.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/ODE.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MonoAdaptiveFixedPointSolver.h>
#include <dolfin/MonoAdaptiveNewtonSolver.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::MonoAdaptiveTimeSlab(ODE& ode)
  : NewTimeSlab(ode), solver(0), adaptivity(ode), nj(0), dofs(0), f(0)
{
  // Choose solver
  solver = chooseSolver();

  // Initialize dofs
  dofs = new real[method->nsize()];

  // Compute the number of dofs
  nj = method->nsize() * N;

  // Initialize values of right-hand side
  const uint fsize = method->qsize() * N;
  f = new real[fsize];
  for (uint j = 0; j < fsize; j++)
    f[j] = 0.0;

  // Initialize solution
  x.init(nj);

  // Evaluate f at initial data for cG(q)
  if ( method->type() == NewMethod::cG )
    ode.f(u0, 0.0, f);
}
//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::~MonoAdaptiveTimeSlab()
{
  if ( solver ) delete solver;
  if ( dofs ) delete [] dofs;
  if ( f ) delete [] f;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab:: build(real a, real b)
{
  cout << "Mono-adaptive time slab: building between "
       << a << " and " << b << endl;

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
  //     << a << " and " << b << endl;

  // Update at t = 0.0
  if ( a < DOLFIN_EPS )
    ode.update(u0, a, false);

  return b;
}
//-----------------------------------------------------------------------------
bool MonoAdaptiveTimeSlab::solve()
{
  dolfin_info("Solving time slab system on [%f, %f].", _a, _b);

  return solver->solve();
}
//-----------------------------------------------------------------------------
bool MonoAdaptiveTimeSlab::shift()
{
  // Get array
  real* xx = x.array();

  // Compute offsets
  const uint foffset = (method->qsize() - 1) * N;
  const uint xoffset = (method->nsize() - 1) * N;

  // Compute f at end-time
  feval(method->qsize() - 1);

  // Compute maximum norm of residual at end-time
  const real k = length();
  real rmax = 0.0;
  for (uint i = 0; i < N; i++)
  {
    // Prepare data for computation of derivative
    const real x0 = u0[i];
    for (uint n = 0; n < method->nsize(); n++)
      dofs[n] = xx[n*N + i];

    // Compute residual
    const real r = fabs(method->residual(x0, dofs, f[foffset + i], k));
    
    // Compute maximum
    if ( r > rmax )
      rmax = r;
  }

  // Compute new time step
  adaptivity.update(rmax, *method);
  
  // Let user update ODE
  const bool end = (_b + DOLFIN_EPS) > ode.T;
  if ( !ode.update(xx + xoffset, _b, end) )
  {
    x.restore(xx);
    return false;
  }

  // Set initial value to end-time value
  for (uint i = 0; i < N; i++)
    u0[i] = xx[xoffset + i];

  // Set f at first quadrature point to f at end-time for cG(q)
  if ( method->type() == NewMethod::cG )
  {
    for (uint i = 0; i < N; i++)
      f[i] = f[foffset + i];
  }

  // Restore array
  x.restore(xx);

  return true;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::sample(real t)
{
  // Do nothing
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
  // FIXME: not implemented

  return 0.0;
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
    cout << " " << f[j];
  cout << endl;
  cout << "----------------------------------------------------------" << endl;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::feval(uint m)
{
  // Evaluation depends on the choice of method
  if ( method->type() == NewMethod::cG )
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
    ode.f(xx + (m-1)*N, t, f + m*N);
    x.restore(xx);
  }
  else
  {
    const real t = _a + method->qpoint(m) * (_b - _a);    
    real* xx = x.array();    
    ode.f(xx + m*N, t, f + m*N);    
    x.restore(xx);
  }
}
//-----------------------------------------------------------------------------
TimeSlabSolver* MonoAdaptiveTimeSlab::chooseSolver()
{
  bool implicit = dolfin_get("implicit");
  std::string solver = dolfin_get("solver");

  if ( solver == "fixed point" )
  {
    if ( implicit )
      dolfin_error("Newton solver must be used for implicit ODE.");

    dolfin_info("Using mono-adaptive fixed point solver.");
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
      dolfin_info("Using mono-adaptive fixed point solver (default for c/dG(q)).");
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

  if ( method->type() == NewMethod::cG )
    return f + N;
  else
    return f;
}
//-----------------------------------------------------------------------------
