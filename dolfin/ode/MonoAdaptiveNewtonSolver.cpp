// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2008-04-22

#include <dolfin/common/real.h>
#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/parameter/parameters.h>
#include <dolfin/la/uBLASKrylovSolver.h>
#include <dolfin/la/UmfpackLUSolver.h>
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
    ts(timeslab), A(timeslab, implicit, piecewise), btmp(0), Mu0(0),
    krylov(0), lu(0), krylov_g(0), lu_g(0)
{
  #ifdef HAS_GMP
  warning("Extended precision monoadaptive Newton solver not implemented. Using double precision");
  #endif


  // Initialize product M*u0 for implicit system
  if (implicit)
  {
    Mu0 = new real[ts.N];
    real_zero(ts.N, Mu0);
  }

  // Choose linear solver
  chooseLinearSolver();
}
//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::~MonoAdaptiveNewtonSolver()
{
  delete [] btmp;
  delete [] Mu0;
  delete krylov;
  delete lu;
  delete krylov_g;
  delete lu_g;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::start()
{
  // Get size of system
  int nj = static_cast<int>(ts.nj);

  // Initialize increment vector
  dx.resize(nj);
  dx.zero();

  // Initialize right-hand side
  b.resize(nj);
  b.zero();
  delete btmp;
  btmp = new real[nj];
  real_zero(nj, btmp);

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
  Feval(btmp);
  for (uint j = 0; j < ts.nj; j++) 
    //Note: Precision lost if working with GMP
    b[j] = to_double(btmp[j]);

  // Solve linear system
  if (krylov)
  {
    const double r = b.norm(linf) + DOLFIN_EPS;
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
  xnorm = real_max_abs(ts.nj, ts.x);

  // Update solution x <- x + dx (note: b = -F)
  for (uint j = 0; j < ts.nj; j++)
    ts.x[j] += dx[j];
  
  // Return maximum increment
  return dx.norm(linf);
}
//-----------------------------------------------------------------------------
dolfin::uint MonoAdaptiveNewtonSolver::size() const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::Feval(real* F)
{
  if (implicit)
    FevalImplicit(F);
  else
    FevalExplicit(F);
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::FevalExplicit(real* F)
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
  real_sub(ts.nj, F, ts.x);
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::FevalImplicit(real* F)
{
  // Use vectors from Jacobian for storing multiplication
  real* xx = A.xx;
  real* yy = A.yy;

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
    message("Using UMFPACK direct solver.");
    lu = new UmfpackLUSolver();
  }
  else
  {
    message("Using uBLAS Krylov solver with no preconditioning.");
    const double ktol = ode.get("ODE discrete Krylov tolerance factor");
    const double _tol = to_double(tol);

    // FIXME: Check choice of tolerances
    krylov = new uBLASKrylovSolver(none);
    krylov->set("Krylov report", monitor);
    krylov->set("Krylov relative tolerance", ktol);
    //Note: Precision lost if working with GMP types
    krylov->set("Krylov absolute tolerance", ktol*_tol);

    message("Using BiCGStab Krylov solver for matrix Jacobian.");
    krylov_g = new KrylovSolver(bicgstab, ilu);
    krylov_g->set("Krylov report", monitor);
    krylov_g->set("Krylov relative tolerance", ktol);
    krylov_g->set("Krylov absolute tolerance", ktol*_tol);
  }
}
//-----------------------------------------------------------------------------
void MonoAdaptiveNewtonSolver::debug()
{
  const uint n = ts.nj;
  uBLASSparseMatrix B(n, n);
  ublas_sparse_matrix& _B = B.mat();
  real* F1 = new real[n];
  real* F2 = new real[n];
  real_zero(n, F1);
  real_zero(n, F2);

  // Iterate over the columns of B
  for (uint j = 0; j < n; j++)
  {
    const real xj = ts.x[j];
    real dx = max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * abs(xj));
		  
    ts.x[j] -= 0.5*dx;
    Feval(F1);

    ts.x[j] = xj + 0.5*dx;
    Feval(F2);
    
    ts.x[j] = xj;

    for (uint i = 0; i < n; i++)
    {
      real dFdx = (F1[i] - F2[i]) / dx;
      if ( abs(dFdx) > DOLFIN_EPS )
        _B(i, j) = to_double(dFdx);
    }
  }

  delete [] F1;
  delete [] F2;

  B.disp();
}
//-----------------------------------------------------------------------------
