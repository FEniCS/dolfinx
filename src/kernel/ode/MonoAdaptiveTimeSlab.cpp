// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/NewMethod.h>
//#include <dolfin/MonoAdaptiveFixedPointSolver.h>
//#include <dolfin/MonoAdaptiveNewtonSolver.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::MonoAdaptiveTimeSlab(ODE& ode)
  : NewTimeSlab(ode), solver(0), nj(0), u0(0)
{
  // Choose solver
  // not implemented

  // Compute the number of dofs
  nj = N * method->nsize();

  // Initialize initial data
  u0 = new real[N];

  // Get initial data
  for (uint i = 0; i < N; i++)
    u0[i] = ode.u0(i);

  // Initialize solution
  u.init(nj);
}
//-----------------------------------------------------------------------------
MonoAdaptiveTimeSlab::~MonoAdaptiveTimeSlab()
{
  if ( u0 ) delete [] u0;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab:: build(real a, real b)
{
  // Copy initial values to solution
  real* x = u.array();
  for (uint n = 0; n < method->nsize(); n++)
  {
    for (uint i = 0; i < N; i++)
    {
      x[n*N + i] = u0[i];
    }
  }
  u.restore(x);

  return 0.0;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::solve()
{
  cout << "Mono-adaptive time slab: solving" << endl;

  //solver->solve();

  cout << "Mono-adaptive time slab: system solved" << endl;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::shift()
{
  // Compute maximum norm of residual at end-time
  // not implemented
  
  // Let user update ODE
  real* x = u.array();
  ode.update(x + (method->nsize() - 1) * N, _b);
  u.restore(x);

  // Set initial value to end-time value (needs to be done last)
  // for (uint i = 0; i < N; i++)
  // u0[i] = u[i];
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::sample(real t)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::usample(uint i, real t)
{
  

  return 0.0;
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::ksample(uint i, real t)
{
  return length();
}
//-----------------------------------------------------------------------------
real MonoAdaptiveTimeSlab::rsample(uint i, real t)
{

  return 0.0;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveTimeSlab::disp() const
{
  

}
//-----------------------------------------------------------------------------
