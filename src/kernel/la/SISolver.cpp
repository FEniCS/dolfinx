// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/basic.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/SISolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SISolver::SISolver()
{
  method = gauss_seidel;
  max_no_iterations = 10000;
  tol = 1.0e-10;
}
//-----------------------------------------------------------------------------
void SISolver::solve(Matrix& A, Vector& x, Vector& b)
{
  // Solve linear system of equations Ax=b, using a stationary iterative (SI) method 
  if ( A.size(0) != A.size(1) ) {
    std::cout << "Must be a square matrix." << std::endl;
    exit(1);
  }
  if ( A.size(0) != b.size() ) {
    std::cout << "Not compatible matrix and vector sizes." << std::endl;
    exit(1);
  }
  if ( x.size() != b.size() ) x.init(b.size());
  
  std::cout << "Using Stationary Iterative Solver solver for linear system of " << b.size() << " unknowns" << std::endl;
  
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
    switch(method){ 
    case richardson:
      iterateRichardson(A,x,b);
      break;
    case jacobi:
      iterateJacobi(A,x,b);
      break;
    case gauss_seidel:
      iterateGaussSeidel(A,x,b);
      break;
    case sor:
      iterateSOR(A,x,b);
      break;
    default:
      std::cout << "Unknown stationary iterative method" << std::endl;
      exit(1);
    }
    
    if (iteration == (max_no_iterations - 1)){
      break;
      //std::cout << "SI iterations did not converge: residual = " << norm_r << std::endl;
      //exit(1);
    }
    norm_r = getResidual(A,x,b);
  }
  
  switch(method){ 
  case richardson:
    std::cout << "Richardson";
    break;
  case jacobi:
    std::cout << "Jacobi";
    break;
  case gauss_seidel:
    std::cout << "Gauss-Seidel";
    break;
  case sor:
    std::cout << "SOR";
    break;
  default:
    std::cout << "Unknown stationary iterative method" << std::endl;
    exit(1);
  }
  if (norm_r < tol*norm_b){
    std::cout << " iterations converged after " << iteration << " iterations (residual = " << norm_r << ")" << std::endl;
  } else{
    std::cout << " iterations did not converge: residual = " << norm_r << std::endl;
  }
}
//-----------------------------------------------------------------------------
void SISolver::setNoSweeps(int max_no_iterations)
{
  this->max_no_iterations = max_no_iterations;
}
//-----------------------------------------------------------------------------
void SISolver::setMethod(SI_method method)
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
