// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/basic.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/SISolver.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
SISolver::SISolver()
{
  method = GAUSS_SEIDEL;
  max_no_iterations = 10000;
  tol = 1.0e-10;
}
//-----------------------------------------------------------------------------
void SISolver::solve(Matrix& A, Vector& x, Vector& b)
{
  // Solve linear system of equations Ax=b, using a stationary iterative (SI) method 
  if ( A.size(0) != A.size(1) )
    dolfin_error("Matrix must be square.");
  if ( A.size(0) != b.size() )
    dolfin_error("Incompatible matrix and vector dimensions.");
  if ( x.size() != b.size() )
	 x.init(b.size());
  
  cout << "Using Stationary Iterative Solver solver for linear system of " << b.size() << " unknowns" << endl;
  
  // Check if b=0 => x=0
  real norm_b = b.norm();
  if (norm_b < DOLFIN_EPS){
    x = 0.0;
    return;
  }
  real norm_r = 2.0*tol*norm_b;
  
  int iteration = 0;
  while (norm_r > tol*norm_b){
    iteration ++;
    switch ( method ) { 
    case RICHARDSON:
      iterateRichardson(A,x,b);
      break;
    case JACOBI:
      iterateJacobi(A,x,b);
      break;
    case GAUSS_SEIDEL:
      iterateGaussSeidel(A,x,b);
      break;
    case SOR:
      iterateSOR(A,x,b);
      break;
    default:
      dolfin_error("Unknown stationary iterative method.");
    }
    
    if (iteration == (max_no_iterations - 1)){
      break;
      //cout << "SI iterations did not converge: residual = " << norm_r << endl;
      //exit(1);
    }
    norm_r = getResidual(A,x,b);
  }
  
  switch ( method ) { 
  case RICHARDSON:
    cout << "Richardson";
    break;
  case JACOBI:
    cout << "Jacobi";
    break;
  case GAUSS_SEIDEL:
    cout << "Gauss-Seidel";
    break;
  case SOR:
    cout << "SOR";
    break;
  default:
    dolfin_error("Unknown stationary iterative method.");
  }
  
  if (norm_r < tol*norm_b)
	 cout << " iterations converged after " << iteration << " iterations (residual = " << norm_r << ")" << endl;
  else
    cout << " iterations did not converge: residual = " << norm_r << endl;
}
//-----------------------------------------------------------------------------
void SISolver::setNoSweeps(int max_no_iterations)
{
  this->max_no_iterations = max_no_iterations;
}
//-----------------------------------------------------------------------------
void SISolver::setMethod(Method method)
{
  this->method = method;
}
//-----------------------------------------------------------------------------
void SISolver::iterateRichardson(Matrix& A, Vector& x, Vector& b)
{
  real aij;
  Vector x0(x);

  int j;
  for (int i=0;i<A.size(0);i++){
    x(i) = b(i);
    for (int pos=0;pos<A.rowSize(i);pos++) {
      aij = A(i,&j,pos);
      if (j == -1) break;
      if (i == j) x(i) += (1.0-aij)*x0(j);
      else x(i) -= aij*x0(j);
    }
  }
}
//-----------------------------------------------------------------------------
void SISolver::iterateJacobi(Matrix& A, Vector& x, Vector& b)
{
  real aii,aij;
  Vector x0(x);

  int j;
  for (int i=0;i<A.size(0);i++){
    x(i) = b(i);
    for (int pos=0;pos<A.rowSize(i);pos++){
      aij = A(i,&j,pos);
      if (j == -1) break;
      if (i==j) aii = aij;
      else x(i) -= aij*x0(j);
    }
    x(i) /= aii;
  }
}
//-----------------------------------------------------------------------------
void SISolver::iterateGaussSeidel(Matrix& A, Vector& x, Vector& b)
{
  real aii,aij;

  int j;
  for (int i=0;i<A.size(0);i++){
    x(i) = b(i);
    for (int pos=0;pos<A.rowSize(i);pos++){
      aij = A(i,&j,pos);
      if (j == -1) break;
      if (i==j) aii = aij;
      else x(i) -= aij*x(j);
    }
    x(i) /= aii;
  }
}
//-----------------------------------------------------------------------------
void SISolver::iterateSOR(Matrix& A, Vector& x, Vector& b)
{
  real aij, aii;
  real omega = 1.0;

  int j;
  for (int i=0;i<A.size(0);i++){
    x(i) = b(i);
    for (int pos=0;pos<A.rowSize(i);pos++){
      aij = A(i,&j,pos);
      if (j == -1 ) break;
      if (j==i){
	aii = aij;
	x(i) += (1.0-omega)*aii*x(i);
      } else{
	x(i) -= omega*aij*x(j);
      }	  
    }
    x(i) /= aii;
  }
}
//-----------------------------------------------------------------------------
real SISolver::getResidual(Matrix &A, Vector &x, Vector &b)
{
  real norm_r = 0.0;
  for (int i=0;i<A.size(0);i++)
    norm_r += sqr(b(i) - A.mult(x,i));

  return sqrt(norm_r);
}
//-----------------------------------------------------------------------------
