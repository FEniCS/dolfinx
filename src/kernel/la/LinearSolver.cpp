// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_math.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/LinearSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void LinearSolver::check(const Matrix& A, Vector& x, const Vector& b) const
{
  if ( A.size(0) != A.size(1) )
    dolfin_error("Matrix must be square.");

  if ( A.size(0) != b.size() )
    dolfin_error("Incompatible dimensions for linear system.");

  if ( x.size() != b.size() )
    x.init(b.size());
}
//-----------------------------------------------------------------------------
real LinearSolver::residual(const Matrix& A, Vector& x, const Vector& b) const
{
  real r = 0.0;
  for (unsigned int i = 0; i < A.size(0); i++)
    r += sqr(b(i) - A.mult(x,i));

  return sqrt(r);
}
//-----------------------------------------------------------------------------
void LinearSolver::iterate(const Matrix& A, Vector& x, const Vector& b,
			   real tol, unsigned int maxiter)
{
  // Check dimensions
  check(A, x, b);

  // Check if b = 0
  real bnorm = b.norm();
  if ( bnorm < DOLFIN_EPS )
  {
    x = 0.0;
    return;
  }

  // Perform iterations
  for (unsigned int n = 0; n < maxiter; n++)
  {
    // Check if the solution has converged
    if ( residual(A, x, b) < tol*bnorm )
      return;

    // Perform one iteration
    iteration(A, x, b);
  }

  // Note that either the iterations converged or we reached the
  // maximum number of iterations
}
//-----------------------------------------------------------------------------
void LinearSolver::iteration(const Matrix& A, Vector& x, const Vector& b)
{
  // This function should be implemented by the sub class
  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
