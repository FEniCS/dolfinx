// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/DenseMatrix.h>
#include <dolfin/Vector.h>
#include <dolfin/DirectSolver.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
void DirectSolver::solve(DenseMatrix& A, Vector& x, const Vector& b) const
{
  lu(A);
  solveLU(A, x, b);
}
//-----------------------------------------------------------------------------
void DirectSolver::hpsolve(const DenseMatrix& A,
			   Vector& x, const Vector& b) const
{
  DenseMatrix LU(A);
  lu(LU);
  hpsolveLU(LU, A, x, b);
}
//-----------------------------------------------------------------------------
void DirectSolver::inverse(DenseMatrix& A, DenseMatrix& Ainv) const
{
  lu(A);
  inverseLU(A, Ainv);
}
//-----------------------------------------------------------------------------
void DirectSolver::lu(DenseMatrix& A) const
{
  // This function replaces the matrix with its LU factorization.
  //
  // Algorithm from "Numerical Recipes in C", except for the following
  // changes:
  //
  //    float replaced by real
  //    removed comments
  //    changed *d to d
  //    changed from [i][j] to [i-1][j-1]
  
  // Check that the matrix is square
  if ( A.size(0) != A.size(1) )
    dolfin_error("Matrix is not square.");
  
  // Prepare the variables for the notation in the algorithm
  real **a = A.values;       // The matrix
  real TINY = 1e-20;           // A small number
  int n = A.size(0);         // Dimension
  real d;                      // Even or odd number of row interchanges
  int *indx = A.permutation; // Permutation of rows
  
  //  void ludcmp(float **a, int n, int *indx, float *d)
  
  // Given a matrix a[1..n][1..n], this routine replaces it by the LU decomposition of a rowwise
  // permutation of itself. a and n are input. a is output, arranged as in equation (2.3.14) above;
  // indx[1..n] is an output vector that records the row permutation effected by the partial
  // pivoting; d is output as +/- 1 depending on whether the number of row interchanges was even
  // or odd, respectively. This routine is used in combination with lubksb to solve linear equations
  // or invert a matrix.
  
  int i,imax,j,k;
  imax = 0;
  real big,dum,sum,temp;
  real* vv = new real[n];
  d = 1.0;
  
  for (i=1;i<=n;i++){
    big=0.0;
    for (j=1;j<=n;j++)
      if ((temp=fabs(a[i-1][j-1])) > big) big=temp;
    if (big == 0.0)
      dolfin_error("Matrix is singular.");
    vv[i-1]=1.0/big;
  }
  
  for (j=1;j<=n;j++){
    for (i=1;i<j;i++){
      sum=a[i-1][j-1];
      for (k=1;k<i;k++) sum -= a[i-1][k-1]*a[k-1][j-1];
      a[i-1][j-1]=sum;
    }
    big=0.0;
    for (i=j;i<=n;i++){
      sum=a[i-1][j-1];
      for (k=1;k<j;k++)
	sum -= a[i-1][k-1]*a[k-1][j-1];
      a[i-1][j-1]=sum;
      if ( (dum=vv[i-1]*fabs(sum)) >= big) {
	big=dum;
	imax=i;
      }
    }
    if (j != imax){ 
      for (k=1;k<=n;k++){
	dum=a[imax-1][k-1];
	a[imax-1][k-1]=a[j-1][k-1];
	a[j-1][k-1]=dum;
      }
      d = -(d);
      vv[imax-1]=vv[j-1]; 
    }
    indx[j-1]=imax;
    if (a[j-1][j-1] == 0.0) a[j-1][j-1]=TINY;
    if (j != n) { 
      dum=1.0/(a[j-1][j-1]);
      for (i=j+1;i<=n;i++) a[i-1][j-1] *= dum;
    }
  }
  
  delete [] vv;
}
//-----------------------------------------------------------------------------
void DirectSolver::solveLU(const DenseMatrix& LU,
			   Vector& x, const Vector& b) const
{
  // Solve the linear system A x = b using a computed LU factorization

  // Check dimensions
  if ( LU.m != LU.n )
    dolfin_error("LU factorization must be a square matrix.");

  if ( LU.m != b.n )
    dolfin_error("Non-matching dimensions for matrix and vector.");

  // Get size
  int n = LU.m;
  
  // Initialize solution vector
  x.init(n);

  // This function solves for the right-hand side b
  //
  // Note! The matrix must be LU factorized first!
  //
  // Algorithm from "Numerical Recipes in C", except for the following
  // changes:
  //
  //    float replaced by real
  //    removed comments
  //    changed from [i][j] to [i-1][j-1]

  // Prepare the variables for the notation in the algorithm
  real **a = LU.values;       // The matrix
  int *indx = LU.permutation; // Permutation
  
  // Copy b to x
  for (int i=0;i<n;i++)
    x.values[i] = b.values[i];
  
  //  void lubksb(float **a, int n, int *indx, float b[])
  
  // Solves the set of n linear equations A X = B. Here a[1..n][1..n] is input, not as the matrix
  // A but rather as its LU decomposition, determined by the routine ludcmp. indx[1..n] is input
  // as the permutation vector returned by ludcmp. b[1..n] is input as the right-hand side vector
  // B, and returns with the solution vector X. a, n, and indx are not modified by this routine
  // and can be left in place for successive calls with different right-hand sides b. This routine takes
  // into account the possibility that b will begin with many zero elements, so it is efficient for use
  // in matrix inversion.
  
  int i,ii=0,ip,j;
  
  float sum;
  
  for (i=1;i<=n;i++){
    ip=indx[i-1];
    sum=x.values[ip-1];
    x.values[ip-1]=x.values[i-1];
    if (ii)	for (j=ii;j<=i-1;j++) sum -= a[i-1][j-1]*x.values[j-1];
    else if (sum) ii=i;
    x.values[i-1]=sum;
  }
  
  for (i=n;i>=1;i--){
    sum=x.values[i-1];
    for (j=i+1;j<=n;j++) sum -= a[i-1][j-1]*x.values[j-1];
    x.values[i-1]=sum/a[i-1][i-1]; 
  }
  
}
//-----------------------------------------------------------------------------
void DirectSolver::inverseLU(const DenseMatrix& LU, DenseMatrix& Ainv) const
{
  // Compute inverse using a computed LU factorization

  // Check dimensions
  if ( LU.m != LU.n )
    dolfin_error("LU factorization must be a square matrix.");

  // Get size
  int n = LU.m;

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
void DirectSolver::hpsolveLU(const DenseMatrix& LU, const DenseMatrix& A,
			     Vector& x, const Vector& b) const
{
  // Solve the linear system A x = b to very high precision, by first
  // computing the inverse using Gaussian elimination, and then using the
  // inverse as a preconditioner for Gauss-Seidel iteration.

  // Check dimensions
  if ( LU.m != LU.n )
    dolfin_error("LU factorization must be a square matrix.");

  if ( A.m != A.n )
    dolfin_error("Matrix must be square.");

  if ( LU.m != b.n )
    dolfin_error("Non-matching dimensions for matrix and vector.");

  if ( LU.m != A.m )
    dolfin_error("Non-matching matrix dimensions.");
  
  // Get size
  int n = LU.m;
  
  // Initialize the solution vector
  x.init(n);

  // Compute product B = Ainv * A
  DenseMatrix B(n, n);
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
  do {
    
    // Gauss-Seidel iteration
    for (int i = 0; i < n; i++) {
      real sum = c(i);
      for (int j = 0; j < n; j++)
	if ( j != i )
	  sum += B(i,j) * x(j);
      x(i) = sum / B(i,i);
    }
    
    // Compute the residual
    res = 0.0;
    for (int i = 0; i < n; i++) {
      real sum = 0.0;
      for (int j = 0; j < n; j++)
	sum += A(i,j) * x(j);
      sum -= b(i);
      res += sum*sum;
    }
    res = sqrt(res);
    
    dolfin_info("residual = %.16e", res);

  } while ( res > DOLFIN_EPS );
  
}
//-----------------------------------------------------------------------------
