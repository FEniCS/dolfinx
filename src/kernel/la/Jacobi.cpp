// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/Jacobi.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Jacobi::Jacobi(const Matrix& A, real tol, unsigned int maxiter)
  : Preconditioner(), A(A), tol(tol), maxiter(maxiter)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Jacobi::~Jacobi()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Jacobi::solve(Vector& x, const Vector& b)
{
  // Perform iterations
  iterate(A, x, b, tol, maxiter);
}
//-----------------------------------------------------------------------------
void Jacobi::solve(const Matrix& A, Vector& x, const Vector& b,
			real tol, unsigned int maxiter)
{
  // Create a Jacobi object
  Jacobi jac(A, tol, maxiter);

  // Solve linear system
  jac.solve(x, b);
}
//-----------------------------------------------------------------------------
void Jacobi::iteration(const Matrix& A, Vector& x, const Vector& b)
{
  // One Jacobi iteration

  real aii = 0.0;
  real aij = 0.0;
  Vector x0(x);
  
  unsigned int j;
  for (unsigned int i = 0; i < A.size(0); i++)
  {
    x(i) = b(i);
    for (unsigned int pos = 0; !A.endrow(i,pos); pos++)
    {
      aij = A(i, j, pos);
      if ( i == j ) 
	aii = aij;
      else 
	x(i) -= aij*x0(j);
    }
    x(i) /= aii;
  }
}
//-----------------------------------------------------------------------------
