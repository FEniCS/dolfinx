// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/Alloc.h>
#include <dolfin/ODE.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>
#include <dolfin/MonoAdaptiveNewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::MonoAdaptiveNewtonSolver
(MonoAdaptiveTimeSlab& timeslab, bool implicit)
  : TimeSlabSolver(timeslab), implicit(implicit),
    piecewise(dolfin_get("matrix piecewise constant")),
    ts(timeslab), A(timeslab, implicit, piecewise), Mu0(0)
{
  // Initialize product M*u0 for implicit system

  if ( implicit )
  {
    Mu0 = new real[ts.N];
    for (uint i = 0; i < ts.N; i++)
       Mu0[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::~MonoAdaptiveNewtonSolver()
{
  if ( Mu0 ) delete [] Mu0;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::start()
{
  // Get size of system
  int nj = static_cast<int>(ts.nj);

  // Initialize increment vector
  dx.init(nj);

  // Initialize right-hand side
  b.init(nj);

  // Initialize Jacobian matrix
  A.init(dx, dx);

  // Recompute Jacobian 
  //A.update();

  // Precompute product M*u0
  if ( implicit )
    ode.M(ts.u0, Mu0, ts.u0, ts.starttime());

  //debug();
  //A.disp();
}
//-----------------------------------------------------------------------------
real MonoAdaptiveNewtonSolver::iteration()
{
  // Evaluate b = -F(x) at current x
  beval();

  //cout << "b = ";
  b.disp();
  
  // Solve linear system F for dx
  solver.solve(A, dx, b);

  //cout << "dx = ";
  dx.disp();
   
  // Get arrays of values for x and dx
  real* xx = ts.x.array();
  real* dxx = dx.array();

  // Update solution x -> x - dx
  for (uint j = 0; j < ts.nj; j++)
    xx[j] += dxx[j];
  
  // Compute maximum increment
  real max_increment = 0.0;
  for (uint j = 0; j < ts.nj; j++)
  {
    const real increment = fabs(dxx[j]);
    if ( increment > max_increment )
      max_increment = increment;
  }

  // Restore arrays
  ts.x.restore(xx);
  dx.restore(dxx);

  return max_increment;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::beval()
{
  if ( implicit )
    bevalImplicit();
  else
    bevalExplicit();

  //cout << "Evaluated right-hand side F:" << endl;
  //b.disp();
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::bevalExplicit()
{
  // Get arrays of values for x and b (assumes uniprocessor case)
  real* bb = b.array();
  real* xx = ts.x.array();

  // Compute size of time step
  const real k = ts.length();

  // Evaluate right-hand side at all quadrature points
  for (uint m = 0; m < method.qsize(); m++)
    ts.feval(m);

  // Update the values at each stage
  for (uint n = 0; n < method.nsize(); n++)
  {
    const uint noffset = n * ts.N;

    // Reset values to initial data
    for (uint i = 0; i < ts.N; i++)
      bb[noffset + i] = ts.u0[i];
    
    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
	bb[noffset + i] += tmp * ts.f[moffset + i];
    }
  }

  // Subtract current values
  for (uint j = 0; j < ts.nj; j++)
    bb[j] -= xx[j];

  // Restore arrays
  b.restore(bb);
  ts.x.restore(xx);
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::bevalImplicit()
{
  // Get arrays of values for x and b (assumes uniprocessor case)
  real* bb = b.array();
  real* xx = ts.x.array();

  // Compute size of time step
  const real a = ts.starttime();
  const real k = ts.length();

  // Evaluate right-hand side at all quadrature points
  for (uint m = 0; m < method.qsize(); m++)
    ts.feval(m);

  // Update the values at each stage
  for (uint n = 0; n < method.nsize(); n++)
  {
    const uint noffset = n * ts.N;

    // Reset values to initial data
    for (uint i = 0; i < ts.N; i++)
      bb[noffset + i] = Mu0[i];
    
    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
	bb[noffset + i] += tmp * ts.f[moffset + i];
    }
  }
  
  // Temporary data array used to store multiplications
  real* z = ts.tmp();

  // Subtract current values (do this after f is used, otherwise
  // we can't use z...)
  for (uint n = 0; n < method.nsize(); n++)
  {
    const uint noffset = n * ts.N;
    if ( piecewise )
    {
      ode.M(xx + noffset, z, ts.u0, a);
    }
    else
    {
      const real t = a + method.npoint(n) * k;
      ode.M(xx + noffset, z, xx + noffset, t);
    }

    for (uint i = 0; i < ts.N; i++)
      bb[noffset + i] -= z[i];
  }

  // Restore arrays
  b.restore(bb);
  ts.x.restore(xx);
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::debug()
{
  const uint n = ts.nj;
  NewMatrix B(n, n);
  NewVector F1(n), F2(n);

  // Iterate over the columns of B
  for (uint j = 0; j < n; j++)
  {
    const real xj = ts.x(j);
    real dx = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * abs(xj));
		  
    ts.x(j) -= 0.5*dx;
    beval();
    F1 = b; // Should be -b

    ts.x(j) = xj + 0.5*dx;
    beval();
    F2 = b; // Should be -b
    
    ts.x(j) = xj;

    for (uint i = 0; i < n; i++)
    {
      real dFdx = (F1(i) - F2(i)) / dx;
      if ( fabs(dFdx) > DOLFIN_EPS )
	B(i, j) = dFdx;
    }
  }

  B.disp();
}
//-----------------------------------------------------------------------------
