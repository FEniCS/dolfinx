// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/basic.h>
#include <dolfin/SISolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SISolver::SISolver()
{
  iterative_method = GAUSS_SEIDEL;
  
  max_no_iterations = 100;

  tol = 1.0e-6;
}
//-----------------------------------------------------------------------------
void SISolver::solve(Matrix& A, Vector& x, Vector& b)
{
  if ( A.size(0) != A.size(1) ) {
	 cout << "Must be a square matrix." << endl;
	 exit(1);
  }

  if ( A.size(0) != b.size() ) {
	 cout << "Not compatible matrix and vector sizes." << endl;
	 exit(1);
  }
  
  if ( x.size() != b.size() )
    x.init(b.size());

  real norm_b = b.norm();
  if ( norm_b < DOLFIN_EPS ) {
	 x = 0.0;
    return;
  }  
  
  residual = 2.0*tol*norm_b;

  iteration = 0;
   while ( residual/norm_b > tol ){
    iteration ++;
    switch( iterative_method ){ 
    case RICHARDSON:
      iterateRichardson(A, x, b);
      break;
    case JACOBI:
      iterateJacobi(A, x, b);
      break;
    case GAUSS_SEIDEL:
      iterateGaussSeidel(A, x, b);
      break;
    case SOR:
      iterateSOR(A, x, b);
      break;
    default:
      cout << "Unknown method" << endl;
		exit(1);
    }
    computeResidual(A,x,b);
  }
  
}
//-----------------------------------------------------------------------------
void SISolver::set(int noit)
{
  max_no_iterations = noit;
}
//-----------------------------------------------------------------------------
void SISolver::set(Method method)
{
  iterative_method = method;
}
//-----------------------------------------------------------------------------
void SISolver::iterateRichardson(SparseMatrix& A, Vector& x, Vector& b)
{
  real aii,aij,norm_b,Ax;
  
  int j;

  Vector x0(x);

  for (int i = 0; i < A.size(0); i++){
    x(i) = 0.0;
    for (int pos = 0; pos < A.rowSize(i); pos++) {
      aij = A(i, &j, pos);
		if ( j == -1 )
		  break;
      if (i==j)
		  x(i) += (1.0-aij)*x0(j);
      else
		  x(i) += -aij*x0(j);
    }
    x(i) += b(i);
  }

}
//-----------------------------------------------------------------------------
void SISolver::iterateJacobi(SparseMatrix& A, Vector& x, Vector& b)
{
  real aii,aij,norm_b,Ax;
  int j;

  Vector x0(x);

  for (int i = 0; i < A.size(0); i++) {
    x(i) = 0.0;
    for (int pos = 0; pos < A.rowSize(i); pos++){
      aij = A(i, &j, pos);
      if ( j == -1 )
		  break;
      if (i==j)
		  aii = aij;
      else
		  x(i) += -aij*x0(j);
    }
    x(i) += b(i);
    x(i) *= 1.0 / aii;
  }
}
//-----------------------------------------------------------------------------
void SISolver::iterateGaussSeidel(SparseMatrix& A, Vector& x, Vector& b)
{
  real aii,aij,Ax;
  
  int j;

  for (int i = 0; i < A.size(0); i++) {
    x(i) = 0.0;
    for (int pos = 0; pos < A.rowSize(i); pos++) {
      aij = A(i, &j, pos);
		if ( j == -1 )
		  break;
      if (j==i)
		  aii = aij;
		else
		  x(i) += -aij*x(j);
    }
    x(i) += b(i);
    x(i) *= 1.0 / aii;
  }

}
//-----------------------------------------------------------------------------
void SISolver::iterateSOR(SparseMatrix& A, Vector& x, Vector& b)
{
  real aii,aij,norm_b,Ax;
  
  int j;

  real omega = 1.0;

  for (int i = 0; i < A.size(0); i++) {
    x(i) = 0.0;
    for (int pos = 0; pos < A.rowSize(i); pos++) {
      aij = A(i, &j, pos);
		if ( j == -1 )
		  break;
      if ( j==i ){
		  aii = aij;
		  x(i) += (1.0-omega)*aii*x(j);
      } else{
		  x(i) += -omega*aij*x(j);
      }	  
    }
    x(i) += b(i);
    x(i) *= 1.0 / aii;
  }
}
//-----------------------------------------------------------------------------
void SISolver::computeResidual(SparseMatrix& A, Vector& x, Vector& b)
{
  residual = 0.0;
  real Axi;
  
  for (int i = 0; i < A.size(0); i++) {
	 Axi = A.mult(x, i);
	 residual += sqr( b(i) - Axi );
  }

  residual = sqrt(residual);
}
//-----------------------------------------------------------------------------
