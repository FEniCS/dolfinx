// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MonoAdaptiveFixedPointSolver.h>
#include <dolfin/MonoAdaptiveNewtonSolver.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::MonoAdaptiveTimeSlab(ODE& ode)
  : NewTimeSlab(ode), solver(0), nj(0), dofs(0), f(0)
{
  // Choose solver
  solver = new MonoAdaptiveFixedPointSolver(*this);
  //solver = new MonoAdaptiveNewtonSolver(*this);

  // Initialize dofs
  dofs = new real[method->nsize()];

  // Compute the number of dofs
  nj = N * method->nsize();

  // Initialize values of right-hand side
  f = new real[nj];
  for (uint j = 0; j < nj; j++)
    f[j] = 0.0;

  // Initialize solution
  x.init(nj);

  // Evaluate f at initial data for cG(q)
  real* xx = x.array();
  if ( method->type() == NewMethod::cG )
    ode.feval(xx, 0.0, f);
  x.restore(xx);
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

  return b;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::solve()
{
  cout << "Mono-adaptive time slab: solving" << endl;

  solver->solve();

  cout << "Mono-adaptive time slab: system solved" << endl;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::shift()
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

  cout << "r = " << rmax << endl;
  
  // Let user update ODE
  ode.update(xx + xoffset, _b);

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
  dolfin_error("Not implemented");
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
    ode.feval(xx + (m-1)*N, t, f + m*N);
    x.restore(xx);
  }
  else
  {
    const real t = _a + method->qpoint(m) * (_b - _a);    
    real* xx = x.array();    
    ode.feval(xx + m*N, t, f + m*N);    
    x.restore(xx);
  }
}
//-----------------------------------------------------------------------------
