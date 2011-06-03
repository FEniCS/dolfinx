// Copyright (C) 2005-2009 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-28
// Last changed: 2009-09-08

#include <dolfin/common/real.h>
#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/uBLASSparseMatrix.h>
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
    piecewise(ode.parameters["matrix_piecewise_constant"]),
    ts(timeslab), A(timeslab, implicit, piecewise), btmp(0), Mu0(ts.N),
    krylov(0), lu(0), krylov_g(0), lu_g(0)
{
  #ifdef HAS_GMP
  warning("Extended precision monoadaptive Newton solver not implemented. Using double precision");
  #endif


  // Initialize product M*u0 for implicit system
//   if (implicit)
//   {
//     Mu0 = new real[ts.N];
//     real_zero(ts.N, Mu0);
//   }

  // Choose linear solver
  chooseLinearSolver();
}
//-----------------------------------------------------------------------------
MonoAdaptiveNewtonSolver::~MonoAdaptiveNewtonSolver()
{
  delete [] btmp;
  //delete [] Mu0;
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
  delete [] btmp;
  btmp = new real[nj];
  real_zero(nj, btmp);

  // Initialize computation of Jacobian
  A.init();

  // Precompute product M*u0
  if (implicit)
    ode.M(ts.u0, Mu0, ts.u0, ts.starttime());
}
//-----------------------------------------------------------------------------
real MonoAdaptiveNewtonSolver::iteration(const real& tol, uint iter,
                                         const real& d0, const real& d1)
{
  // Evaluate b = -F(x) at current x
  Feval(btmp);
  for (uint j = 0; j < ts.nj; j++)
    //Note: Precision lost if working with GMP
    b[j] = to_double(btmp[j]);

  // Solve linear system
  if (krylov)
  {
    const double r = b.norm("linf") + to_double( real_epsilon() );
    b /= r;
    num_local_iterations += krylov->solve(A, dx, b);
    dx *= r;
  }
  else
  {
    // FIXME: Implement a better check
    if (d1 >= 0.5*d0)
      A.update();
    lu->set_operator(A.matrix());
    lu->solve(dx, b);
  }

  // Save norm of old solution
  xnorm = real_max_abs(ts.nj, ts.x);

  // Update solution x <- x + dx (note: b = -F)
  for (uint j = 0; j < ts.nj; j++)
    ts.x[j] += dx[j];

  // Return maximum increment
  return dx.norm("linf");
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
  Array<real> xx(ts.N, A.xx.data());
  Array<real> yy(ts.N, A.yy.data());

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
    ts.copy(ts.x, noffset, xx.data().get(), 0, ts.N);

    // Do multiplication
    if ( piecewise )
    {
      ode.M(xx, yy, ts.u0, a);
    }
    else
    {
      const real t = a + method.npoint(n) * k;
      ts.copy(ts.x, noffset, ts.u);
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
  const std::string linear_solver = ode.parameters["linear_solver"];

  // First determine if we should use a direct solver
  bool direct = false;
  if ( linear_solver == "direct" )
    direct = true;
  else if ( linear_solver == "iterative" )
    direct = false;
  else if ( linear_solver == "auto" )
  {
    /*
    const uint ode_size_threshold = ode.parameters("size_threshold");
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
    info("Using UMFPACK direct solver.");
    lu = new UmfpackLUSolver();
  }
  else
  {
    info("Using uBLAS Krylov solver with no preconditioning.");
    const double ktol = ode.parameters["discrete_krylov_tolerance_factor"];
    const double _tol = to_double(tol);

    // FIXME: Check choice of tolerances
    krylov = new uBLASKrylovSolver("default", "none");
    krylov->parameters["report"] = monitor;
    krylov->parameters["relative_tolerance"] = ktol;
    //Note: Precision lost if working with GMP types
    krylov->parameters["absolute_tolerance"] = ktol*_tol;

    info("Using BiCGStab Krylov solver for matrix Jacobian.");
    krylov_g = new KrylovSolver("bicgstab", "ilu");
    krylov_g->parameters["report"] = monitor;
    krylov_g->parameters["relative_tolerance"] = ktol;
    krylov_g->parameters["absolute_tolerance"] = ktol*_tol;
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
    real dx = real_max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * real_abs(xj));

    ts.x[j] -= 0.5*dx;
    Feval(F1);

    ts.x[j] = xj + 0.5*dx;
    Feval(F2);

    ts.x[j] = xj;

    for (uint i = 0; i < n; i++)
    {
      real df_dx = (F1[i] - F2[i]) / dx;
      if ( real_abs(df_dx) > real_epsilon() )
        _B(i, j) = to_double(df_dx);
    }
  }

  delete [] F1;
  delete [] F2;

  info(B);
}
//-----------------------------------------------------------------------------
