// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/GaussSeidel.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GaussSeidel::GaussSeidel(const Matrix& A, real tol, unsigned int maxiter)
  : Preconditioner(), A(A), tol(tol), maxiter(maxiter)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GaussSeidel::~GaussSeidel()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GaussSeidel::solve(Vector& x, const Vector& b)
{
  // Perform iterations
  iterate(A, x, b, tol, maxiter);
}
//-----------------------------------------------------------------------------
void GaussSeidel::solve(const Matrix& A, Vector& x, const Vector& b,
			real tol, unsigned int maxiter)
{
  // Create a Gauss-Seidel object
  GaussSeidel gs(A, tol, maxiter);

  // Solve linear system
  gs.solve(x, b);
}
//-----------------------------------------------------------------------------
void GaussSeidel::iteration(const Matrix& A, Vector& x, const Vector& b)
{
  // One Gauss-Seidel iteration

  real aii = 0.0;
  real aij = 0.0;
  
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
	x(i) -= aij*x(j);
    }
    x(i) /= aii;
  }
}
//-----------------------------------------------------------------------------
