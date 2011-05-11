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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-27
// Last changed: 2009-09-08

#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/la/uBLASSparseMatrix.h>
#include <dolfin/la/uBLASKrylovSolver.h>
#include "Alloc.h"
#include "ODE.h"
#include "Method.h"
#include "MultiAdaptiveTimeSlab.h"
#include "MultiAdaptiveJacobian.h"
#include "UpdatedMultiAdaptiveJacobian.h"
#include "MultiAdaptiveNewtonSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptiveNewtonSolver::MultiAdaptiveNewtonSolver
(MultiAdaptiveTimeSlab& timeslab)
  : TimeSlabSolver(timeslab), ts(timeslab), A(0),
    mpc(timeslab, method), solver(new uBLASKrylovSolver(mpc)),
    f(0), u(0), num_elements(0), num_elements_mono(0),
    updated_jacobian(ode.parameters["updated_jacobian"])
{
  // Initialize local arrays
  f = new real[method.qsize()];
  u = new real[method.nsize()];

  // Set parameters for Krylov solver
  assert(solver);
  solver->parameters["report"] = monitor;
  solver->parameters["absolute_tolerance"] = 0.01;
  solver->parameters["relative_tolerance"] = 0.01 * to_double(tol);

  // Initialize Jacobian
  if ( updated_jacobian )
    A = new UpdatedMultiAdaptiveJacobian(*this, timeslab);
  else
    A = new MultiAdaptiveJacobian(*this, timeslab);
}
//-----------------------------------------------------------------------------
MultiAdaptiveNewtonSolver::~MultiAdaptiveNewtonSolver()
{
  // Compute multi-adaptive efficiency index
  if ( num_elements > 0 )
  {
    const real alpha = num_elements_mono / static_cast<real>(num_elements);
    info("Multi-adaptive efficiency index: %.3f", to_double(alpha));
  }

  // Delete local arrays
  if (f)
    delete [] f;
  if (u)
    delete [] u;

  // Delete Jacobian
  delete A;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::end()
{
  num_elements += ts.ne;
  num_elements_mono += ts.length() / ts.kmin * static_cast<real>(ts.ode.size());
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::start()
{
  // Get size of system
  int nj = static_cast<int>(ts.nj);

  // Initialize increment vector
  dx.resize(nj);
  dx.zero();

  // Initialize right-hand side
  b.resize(nj);
  b.zero();

  // Recompute Jacobian on each time slab
  A->init();

  //debug();
  //info(A);
}
//-----------------------------------------------------------------------------
real MultiAdaptiveNewtonSolver::iteration(const real& tol, uint iter, const real& d0, const real& d1)
{
  assert(solver);

  // Evaluate b = -F(x) at current x
  Feval(b);

  // FIXME: Scaling needed for PETSc Krylov solver, but maybe not for uBLAS?

  // Save norm of old solution
  xnorm = 0.0;
  for (uint j = 0; j < ts.nj; j++)
    xnorm = real_max(xnorm, real_abs(ts.jx[j]));

  // Solve linear system
  const double r = b.norm("linf") + to_double( real_epsilon() );
  b /= r;
  num_local_iterations += solver->solve(*A, dx, b);
  dx *= r;

  // Update solution x -> x + dx (note: b = -F)
  for (uint j = 0; j < ts.nj; j++)
    ts.jx[j] += dx[j];

  // Compute maximum increment
  real max_increment = 0.0;
  for (uint j = 0; j < ts.nj; j++)
  {
    const real increment = real_abs(dx[j]);
    if (increment > max_increment)
      max_increment = increment;
  }

  return max_increment;
}
//-----------------------------------------------------------------------------
dolfin::uint MultiAdaptiveNewtonSolver::size() const
{
  return ts.nj;
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::Feval(uBLASVector& F)
{
  // Reset dof
  uint j = 0;

  // Reset current sub slab
  int s = -1;

  // Reset elast
  for (uint i = 0; i < ts.N; i++)
    ts.elast[i] = -1;

  // Iterate over all elements
  for (uint e = 0; e < ts.ne; e++)
  {
    // Cover all elements in current sub slab
    s = ts.cover_next(s, e);

    // Get element data
    const uint i = ts.ei[e];
    const real a = ts.sa[s];
    const real b = ts.sb[s];
    const real k = b - a;

    // Get initial value for element
    const int ep = ts.ee[e];
    const real x0 = ( ep != -1 ? ts.jx[ep*method.nsize() + method.nsize() - 1] : ts.u0[i] );

    // Evaluate right-hand side at quadrature points of element
    if ( method.type() == Method::cG )
      ts.cg_feval(f, s, e, i, a, b, k);
    else
      ts.dg_feval(f, s, e, i, a, b, k);

    // Update values on element using fixed-point iteration
    method.update(x0, f, k, u);

    // Subtract current values
    for (uint n = 0; n < method.nsize(); n++)
      F[j + n] = to_double(u[j] - ts.jx[j + n]);

    // Update dof
    j += method.nsize();
  }
}
//-----------------------------------------------------------------------------
void MultiAdaptiveNewtonSolver::debug()
{
  const uint n = ts.nj;
  uBLASSparseMatrix B(n, n);
  uBLASVector F1(n), F2(n);
  ublas_sparse_matrix& _B = B.mat();

  // Iterate over the columns of B
  for (uint j = 0; j < n; j++)
  {
    const real xj = ts.jx[j];
    real dx = real_max(DOLFIN_SQRT_EPS, DOLFIN_SQRT_EPS * real_abs(xj));

    ts.jx[j] -= 0.5*dx;
    Feval(F1);

    ts.jx[j] = xj + 0.5*dx;
    Feval(F2);

    ts.jx[j] = xj;

    for (uint i = 0; i < n; i++)
    {
      real df_dx = (F1[i] - F2[i]) / dx;
      if (real_abs(df_dx) > real_epsilon())
        _B(i, j) = to_double(df_dx);
    }
  }

  info(B);
}
//-----------------------------------------------------------------------------
