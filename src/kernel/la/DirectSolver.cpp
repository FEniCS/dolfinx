// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Niklas Ericsson 2003: lu() and lusolve()

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/DirectSolver.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
void DirectSolver::solve(Matrix& A, Vector& x, const Vector& b) const
{
  check(A);
  lu(A);
  solveLU(A, x, b);
}
//-----------------------------------------------------------------------------
void DirectSolver::hpsolve(const Matrix& A,
			   Vector& x, const Vector& b) const
{
  check(A);
  Matrix LU(A);
  lu(LU);
  hpsolveLU(LU, A, x, b);
}
//-----------------------------------------------------------------------------
void DirectSolver::inverse(Matrix& A, Matrix& Ainv) const
{
  check(A);
  lu(A);
  inverseLU(A, Ainv);
}
//-----------------------------------------------------------------------------
void DirectSolver::lu(Matrix& A) const
{  
  // Check that the matrix is dense
  check(A);
  
  // Check that the matrix is square
  if ( A.size(0) != A.size(1) )
    dolfin_error("Matrix is not square.");
  
  // Initialize permutation
  A.initperm();

  // Get data from matrix
  int n      = A.size(0);
  real** a   = A.values();
  int* index = A.permutation();
  
  // Compute the LU factorization
  for (int j = 0; j < (n-1); j++) {
    
    // Partial pivoting, find the largest element on and below the diagonal
    int imax = j;    
    for (int i = j+1; i < n; i++)
      if ( fabs(a[i][j]) > fabs(a[imax][j]) )
	imax = i;
    
    // Get diagonal element
    real diag = a[imax][j];
    if ( fabs(diag) < DOLFIN_EPS )
      dolfin_error("Matrix is singular to working precision.");
    
    // Swap permutation vector
    int tmp1 = index[j];
    index[j] = index[imax];
    index[imax] = tmp1;

    // Swap matrix rows
    real* tmp2 = a[j];
    a[j] = a[imax];
    a[imax] = tmp2;
    
    // Gaussian elimination
    for (int i = j+1; i < n; i++) {
      real alpha = a[i][j] / diag;
      a[i][j] = alpha;
      for (int k = j+1; k < n; k++)
	a[i][k] -= alpha * a[j][k];
    }

  }
}
//-----------------------------------------------------------------------------
void DirectSolver::solveLU(const Matrix& LU, Vector& x, const Vector& b) const
{
  // Check that the matrix is dense
  check(LU);
  
  // Check dimensions
  if ( LU.size(0) != LU.size(1) )
    dolfin_error("LU factorization must be a square matrix.");
  if ( LU.size(0) != b.size() )
    dolfin_error("Non-matching dimensions for matrix and vector.");

  // Get data from matrix
  int n = LU.size(0);
  real** const a   = LU.values();
  int* const index = LU.permutation();
  
  // Check that the matrix has been LU factorized
  if ( !index )
    dolfin_error("Matrix is not LU factorized.");

  // Initialize solution vector
  x.init(n);

  // First solve Lx = b by forward substitution
  for (int i = 0; i < n; i++) {
    real sum = b(index[i]);
    for (int j = 0; j < i; j++)
      sum -= a[i][j] * x(j);
    x(i) = sum;
  }
 
  // Then solve Ux = x by backsubstitution
  for (int i = n-1; i >= 0; i--) {
    real sum = x(i);
    for (int j = i+1; j < n; j++)
      sum -= a[i][j] * x(j);
    x(i) = sum / a[i][i];
  }
}
//-----------------------------------------------------------------------------
void DirectSolver::inverseLU(const Matrix& LU, Matrix& Ainv) const
{
  check(LU);
  check(Ainv);
 
  // Compute inverse using a computed LU factorization
 
  // Check dimensions
  if ( LU.size(0) != LU.size(1) )
    dolfin_error("LU factorization must be a square matrix.");
 
  // Get size
  int n = LU.size(0);
 
  // Initialize inverse
  Ainv.init(n, n);
 
  // Unit vector and solution
  Vector e(n);
  Vector x;
 
  // Compute inverse
  for (int j = 0; j < n; j++) {
                                                                                                                                                            
    e(j) = 1.0;
    solveLU(LU, x, e);
    e(j) = 0.0;
                                                                                                                                                            
    for (int i = 0; i < n; i++)
      Ainv(i, j) = x(i);
                                                                                                                                                            
  }
}
//-----------------------------------------------------------------------------
void DirectSolver::hpsolveLU(const Matrix& LU, const Matrix& A,
			     Vector& x, const Vector& b) const
{
  // Solve the linear system A x = b to very high precision, by first
  // computing the inverse using Gaussian elimination, and then using the
  // inverse as a preconditioner for Gauss-Seidel iteration.
  //
  // This is probably no longer needed (but was needed before when the
  // LU factorization did not give full precision because of a float lying
  // around). Now the improvement compared to solveLU() is not astonishing.
  // Typically, the residual will decrease from about 3e-17 so say 1e-17.
  // Sometimes it will even increase.

  // Check that matrices are dense
  check(LU);
  check(A);

  // Check dimensions
  if ( LU.size(0) != LU.size(1) )
    dolfin_error("LU factorization must be a square matrix.");

  if ( A.size(0) != A.size(1) )
    dolfin_error("Matrix must be square.");

  if ( LU.size(0) != b.size() )
    dolfin_error("Non-matching dimensions for matrix and vector.");

  if ( LU.size(0) != A.size(1) )
    dolfin_error("Non-matching matrix dimensions.");
  
  // Get size
  int n = LU.size(0);
  
  // Start with the solution from LU factorization
  solveLU(LU, x, b);

  // Compute product B = Ainv * A
  Matrix B(n, n);
  Vector colA(n);
  Vector colB(n);
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++)
      colA(i) = A(i,j);
    solveLU(LU, colB, colA);
    for (int i = 0; i < n; i++)
      B(i,j) = colB(i);
  }

  // Compute product c = Ainv * b
  Vector c(n);
  solveLU(LU, c, b);

  // Solve B x = c using Gauss-Seidel iteration
  real res = 0.0;
  while ( true ) {

    // Compute the residual
    res = 0.0;
    for (int i = 0; i < n; i++)
      res += sqr(A.mult(x,i) - b(i));
    res /= real(n);
    res = sqrt(res);

    // Check residual
    if ( res < DOLFIN_EPS )
      break;
    
    // Gauss-Seidel iteration
    for (int i = 0; i < n; i++) {
      real sum = c(i);
      for (int j = 0; j < n; j++)
	if ( j != i )
	  sum -= B(i,j) * x(j);
      x(i) = sum / B(i,i);
    }
    
  }
  
}
//-----------------------------------------------------------------------------
void DirectSolver::check(const Matrix& A) const
{
  if ( A.type() != Matrix::dense )
    dolfin_error("Matrix must be dense to use the direct solver. Consider using dense().");
}
//-----------------------------------------------------------------------------
