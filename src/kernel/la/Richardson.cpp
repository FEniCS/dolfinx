// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/Richardson.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Richardson::Richardson(const Matrix& A, real tol, unsigned int maxiter)
  : Preconditioner(), A(A), tol(tol), maxiter(maxiter)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Richardson::~Richardson()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Richardson::solve(Vector& x, const Vector& b)
{
  // Perform iterations
  iterate(A, x, b, tol, maxiter);
}
//-----------------------------------------------------------------------------
void Richardson::solve(const Matrix& A, Vector& x, const Vector& b,
			real tol, unsigned int maxiter)
{
  // Create a Richardson object
  Richardson jac(A, tol, maxiter);
    
  // Solve linear system
  jac.solve(x, b);
}
//-----------------------------------------------------------------------------
void Richardson::iteration(const Matrix& A, Vector& x, const Vector& b)
{
  // One Richardson iteration

  real aij;
  Vector x0(x);

  unsigned int j;
  for (unsigned int i=0;i<A.size(0);i++)
  {
    x(i) = b(i);
    for (unsigned int pos = 0; !A.endrow(i,pos); pos++) 
    {
      aij = A(i,j,pos);
      if (i == j) 
	x(i) += (1.0-aij)*x0(j);
      else 
	x(i) -= aij*x0(j);
    }
  }
}
//-----------------------------------------------------------------------------
