// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2008-04-22

#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/parameter/parameters.h>
#include <dolfin/la/uBlasKrylovSolver.h>
#include <dolfin/la/uBlasLUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/LUSolver.h>
#include "Alloc.h"
#include "ODE.h"
#include "Method.h"
#include "MonoAdaptiveTimeSlab.h"
#include "MonoAdaptiveNewtonSolver.h"
#include <dolfin/common/timing.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::MonoAdaptiveNewtonSolver
(MonoAdaptiveTimeSlab& timeslab, bool implicit)
  : TimeSlabSolver(timeslab), implicit(implicit),
    piecewise(ode.get("ODE matrix piecewise constant")),
    ts(timeslab), A(timeslab, implicit, piecewise),
    krylov(0), lu(0), krylov_g(0), lu_g(0)
{
  // Initialize product M*u0 for implicit system
  if ( implicit )
  {
    Mu0.init(ts.N);
    Mu0.zero();
  }

  // Choose linear solver
  chooseLinearSolver();
}
//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::~MonoAdaptiveNewtonSolver()
{
  if ( krylov )
    delete krylov;
  if ( lu )
    delete lu;
  if ( krylov_g )
    delete krylov_g;
  if ( lu_g )
    delete lu_g;
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

  // Initialize computation of Jacobian
  A.init();

  // Precompute product M*u0
  if (implicit)
    ode.M(ts.u0, Mu0, ts.u0, ts.starttime());

  //debug();
  //A.disp(true, 10);
}
//-----------------------------------------------------------------------------
real MonoAdaptiveNewtonSolver::iteration(real tol, uint iter, real d0, real d1)
{
  // Evaluate b = -F(x) at current x
  Feval(b);

  if (krylov)
  {
    const real r = b.norm(linf) + DOLFIN_EPS;
    b /= r;
    num_local_iterations += krylov->solve(A, dx, b);
    dx *= r;
  }
  else
  {
    // FIXME: Implement a better check
    if (d1 >= 0.5*d0)
      A.update();
    lu->solve(A.matrix(), dx, b);
  }

  // Save norm of old solution
  xnorm = ts.x.norm(linf);

  // Update solution x <- x + dx (note: b = -F)
  ts.x += dx;
  
  // Return maximum increment
  return dx.norm(linf);
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
      F[noffset + i] = ts.u0[i];
    
    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
        F[noffset + i] += tmp * ts.fq[moffset + i];
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
      F[noffset + i] = Mu0[i];
    
    // Add weights of right-hand side
    for (uint m = 0; m < method.qsize(); m++)
    {
      const real tmp = k * method.nweight(n, m);
      const uint moffset = m * ts.N;
      for (uint i = 0; i < ts.N; i++)
        F[noffset + i] += tmp * ts.fq[moffset + i];
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
      ode.M(xx, yy, ts.u0, a);
    }
    else
    {
      const real t = a + method.npoint(n) * k;
      ts.copy(ts.x, noffset, ts.u, 0, ts.N);
      ode.M(xx, yy, ts.u, t);
    }

    // Copy values from yy
    for (uint i = 0; i < ts.N; i++)
      F[noffset + i] -= yy[i];
  }
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::chooseLinearSolver()
{
  const std::string linear_solver = ode.get("ODE linear solver");
  
  // First determine if we should use a direct solver
  bool direct = false;  
  if ( linear_solver == "direct" )
    direct = true;
  else if ( linear_solver == "iterative" )
    direct = false;
  else if ( linear_solver == "auto" )
  {
    /*
    const uint ode_size_threshold = ode.get("ODE size threshold");
    if ( ode.size() > ode_size_threshold )
      direct = false;
    else
      direct = true;
    */

    // FIXME: Seems to be a bug (check stiff demo)
    // so we go with the iterative solver for now
    direct = false;
  }

  // Initialize linear solver
  if ( direct )
  {
    message("Using uBlas direct solver.");
    lu = new uBlasLUSolver();
  }
  else
  {
    message("Using uBlas Krylov solver with no preconditioning.");
    const real ktol = ode.get("ODE discrete Krylov tolerance factor");

    // FIXME: Check choice of tolerances
    krylov = new uBlasKrylovSolver(none);
    krylov->set("Krylov report", monitor);
    krylov->set("Krylov relative tolerance", ktol);
    krylov->set("Krylov absolute tolerance", ktol*tol);

    message("Using BiCGStab Krylov solver for matrix Jacobian");
    krylov_g = new KrylovSolver(bicgstab, ilu);
    krylov_g->set("Krylov report", monitor);
    krylov_g->set("Krylov relative tolerance", ktol);
    krylov_g->set("Krylov absolute tolerance", ktol*tol);
  }
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::debug()
{
  const uint n = ts.nj;
  uBlasSparseMatrix B(n, n);
  ublas_sparse_matrix& _B = B.mat();
  uBlasVector F1(n), F2(n);

  // Iterate over the columns of B
  for (uint j = 0; j < n; j++)
  {
    const real xj = ts.x[j];
    real dx = std::max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * std::abs(xj));
		  
    ts.x[j] -= 0.5*dx;
    Feval(F1);

    ts.x[j] = xj + 0.5*dx;
    Feval(F2);
    
    ts.x[j] = xj;

    for (uint i = 0; i < n; i++)
    {
      real dFdx = (F1[i] - F2[i]) / dx;
      if ( fabs(dFdx) > DOLFIN_EPS )
        _B(i, j) = dFdx;
    }
  }

  B.disp();
}
//-----------------------------------------------------------------------------
