// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/SOR.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SOR::SOR(const Matrix& A, real tol, unsigned int maxiter)
  : Preconditioner(), A(A), tol(tol), maxiter(maxiter)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SOR::~SOR()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void SOR::solve(Vector& x, const Vector& b)
{
  // Perform iterations
  iterate(A, x, b, tol, maxiter);
}
//-----------------------------------------------------------------------------
void SOR::solve(const Matrix& A, Vector& x, const Vector& b,
			real tol, unsigned int maxiter)
{
  // Create a SOR object
  SOR jac(A, tol, maxiter);

  // Solve linear system
  jac.solve(x, b);
}
//-----------------------------------------------------------------------------
void SOR::iteration(const Matrix& A, Vector& x, const Vector& b)
{
  // One SOR iteration

  real aij = 0.0;
  real aii = 0.0;
  real omega = 1.5;
  Vector x0(x);

  unsigned int j;
  for (unsigned int i = 0; i < A.size(0); i++) 
  {
    x(i) = omega * b(i);
    for (unsigned int pos = 0; !A.endrow(i,pos); pos++) {
      aij = A(i,j,pos);
      if (j==i)
      {
	aii = aij;
	x(i) += (1.0-omega)*aii*x0(i);
      } 
      else
      {
	x(i) -= omega*aij*x(j);
      }	  
    }
    x(i) /= aii;
  }
}
//-----------------------------------------------------------------------------
