// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-28
// Last changed: 2006-07-06

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/ParameterSystem.h>
#include <dolfin/Alloc.h>
#include <dolfin/ODE.h>
#include <dolfin/Method.h>
#include <dolfin/MonoAdaptiveTimeSlab.h>
#include <dolfin/MonoAdaptiveNewtonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::MonoAdaptiveNewtonSolver
(MonoAdaptiveTimeSlab& timeslab, bool implicit)
  : TimeSlabSolver(timeslab), implicit(implicit),
    piecewise(get("ODE matrix piecewise constant")),
    ts(timeslab), A(timeslab, implicit, piecewise)
{
  // Initialize product M*u0 for implicit system
  if ( implicit )
  {
    Mu0.init(ts.N);
    Mu0 = 0.0;
  }

  // Initialize linear solver
  const real ktol = get("ODE discrete Krylov tolerance factor");
  dolfin_info("Using uBlas Krylov solver with no preconditioning.");
  solver.set("Krylov report", monitor);
  solver.set("Krylov relative tolerance", ktol);
  solver.set("Krylov absolute tolerance", ktol*tol); // FIXME: Is this a good choice?
}
//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::~MonoAdaptiveNewtonSolver()
{
  // Do nothing
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

  // Recompute Jacobian
  //A.update();

  // Precompute product M*u0
  if ( implicit )
  {
    ts.copy(ts.u0, 0, ts.u, 0, ts.N);
    ode.M(ts.u, Mu0, ts.u, ts.starttime());
  }

  //debug();
  //A.disp(true, 10);
}
//-----------------------------------------------------------------------------
real MonoAdaptiveNewtonSolver::iteration(uint iter, real tol)
{
  // Evaluate b = -F(x) at current x
  Feval(b);

  // FIXME: Scaling needed for PETSc Krylov solver, but maybe not for uBlas?

  //cout << "A = ";
  //A.disp(10);
  //cout << "b = ";
  //b.disp();

  // Solve linear system
  const real r = b.norm(uBlasVector::linf) + DOLFIN_EPS;
  b /= r;
  num_local_iterations += solver.solve(A, dx, b, pc);
  dx *= r;

  //cout << "A = "; A.disp(10);
  //cout << "b = "; b.disp();
  //cout << "dx = "; dx.disp();

  // Update solution x <- x + dx (note: b = -F)
  ts.x += dx;
  
  // Return maximum increment
  return dx.norm(uBlasVector::linf);
}
//-----------------------------------------------------------------------------
dolfin::uint MonoAdaptiveNewtonSolver::size() const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::Feval(uBlasVector& F)
{
  if ( implicit )
    FevalImplicit(F);
  else
    FevalExplicit(F);
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::FevalExplicit(uBlasVector& F)
{
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
      F(noffset + i) = ts.u0[i];
    
    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
	F(noffset + i) += tmp * ts.fq[moffset + i];
    }
  }

  // Subtract current values
  F -= ts.x;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::FevalImplicit(uBlasVector& F)
{
  // Use vectors from Jacobian for storing multiplication
  uBlasVector& xx = A.xx;
  uBlasVector& yy = A.yy;

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
      F(noffset + i) = Mu0(i);
    
    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
	F(noffset + i) += tmp * ts.fq[moffset + i];
    }
  }
  
  // Subtract current values
  for (uint n = 0; n < method.nsize(); n++)
  {
    const uint noffset = n * ts.N;

    // Copy values to xx
    ts.copy(ts.x, noffset, xx, 0, ts.N);

    // Do multiplication
    if ( piecewise )
    {
      ts.copy(ts.u0, 0, ts.u, 0, ts.N);
      ode.M(xx, yy, ts.u, a);
    }
    else
    {
      const real t = a + method.npoint(n) * k;
      ts.copy(ts.x, noffset, ts.u, 0, ts.N);
      ode.M(xx, yy, ts.u, t);
    }

    // Copy values from yy
    for (uint i = 0; i < ts.N; i++)
      F(noffset + i) -= yy[i];
  }
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::debug()
{
  const uint n = ts.nj;
  uBlasSparseMatrix B(n, n);
  uBlasVector F1(n), F2(n);

  // Iterate over the columns of B
  for (uint j = 0; j < n; j++)
  {
    const real xj = ts.x(j);
    real dx = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * std::abs(xj));
		  
    ts.x(j) -= 0.5*dx;
    Feval(F1);

    ts.x(j) = xj + 0.5*dx;
    Feval(F2);
    
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
