// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>

#include <dolfin/basic.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/DenseMatrix.h>
#include <dolfin/SISolver.h>
#include <dolfin/KrylovSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver()
{
  method = GMRES;
  pc     = NONE;
  tol    = 1.0e-10;
}
//-----------------------------------------------------------------------------
void KrylovSolver::set(Method method)
{
  this->method = method;
}
//-----------------------------------------------------------------------------
void KrylovSolver::solve(Matrix &A, Vector &x, Vector &b)
{
  // Write a message
  cout << "Solving linear system for " << b.size() << " unknowns, using Krylov solver." << endl;
  
  // Check if we need to resize x
  if ( x.size() != b.size() ) x.init(b.size());

  // Check if b = 0
  if ( b.norm() < DOLFIN_EPS ){
    cout << "b = 0, so x = 0" << endl;
    x = 0.0;
    return;
  }
  
  // Choose method
  switch( method ){ 
  case GMRES:
    solveGMRES(A,x,b);
    break;
  case CG:
    solveCG(A,x,b);
    break;
  default:
    cout << "KrylovSolver::Solve(): Krylov method not implemented" << endl;
  }
}
//-----------------------------------------------------------------------------
void KrylovSolver::solveGMRES(Matrix &A, Vector &x, Vector &b)
{
  cout << "Using the GMRES solver." << endl;
  
  int k_max           = 20;  // no iterations before restart
  int max_no_restarts = 100; // max no restarts

  real norm_residual = residual(A, x, b);
  
  for (int i = 0;i < max_no_restarts; i++){
    
    restartGMRES(A, x, b, k_max);
    norm_residual = residual(A, x, b);

    if ( norm_residual < (tol*norm_b) ){
      if ( i > 0 ) cout << "Restarted GMRES converged after " << i*k_max+no_iterations << " iterations";
      else cout << "GMRES converged after " << no_iterations << " iterations";
      
      cout << " (residual = " << norm_residual << ")." << endl;
      break;
    }
	 
    if (i == max_no_restarts-1) {
      cout << "GMRES iterations did not converge: residual = " << norm_residual << endl;
      exit(1);
    }
  }

}
//-----------------------------------------------------------------------------
real KrylovSolver::restartGMRES(Matrix &A, Vector &x, Vector &b, int k_max)
{
  // Solve preconditioned problem Ax = AP^(-1)Px = b, 
  // by first solving AP^(-1)u=b through GMRES, then 
  // solve Px = u to get x. Extrem cases is P = A (changes nothing) 
  // and P = I (corresponds to no preconditioner). 
  // AP^(-1)u=b is solved to a tolerance rho/|b| < tol, 
  // where rho is the norm of the residual b-AP^(-1)u = b-Ax,
  // at a maximal number of iterations k_max, 
  // starting from startvector u = Px. 

  // no unknowns
  int n = x.size();
  
  // initialization of temporary variables 
  DenseMatrix mat_H(k_max+1,k_max+1);
  DenseMatrix mat_r(k_max+1,k_max+1);
  DenseMatrix mat_v(n,k_max+1);

  Vector vec_s(k_max+1); 
  Vector vec_y(k_max+1); 
  Vector vec_w(k_max+1); 
  Vector vec_c(k_max+1); 
  Vector vec_g(k_max+1); 

  Vector tmpvec(n);
  Vector vec_tmp1(A.size(0)); 
  Vector vec_tmp2(A.size(0)); 

  real tmp,tmp1,tmp2,nu,norm_v,htmp;

  // norm of rhs b
  real norm_b = b.norm();
  
  // Compute start residual = b-AP^(-1)u = b-Ax.
  Vector r(n);
  residual(A,x,b,r);
  real norm_residual = r.norm();

  // set start column of v to residual/|residual|
  for (int i = 0; i <n; i++)
	 mat_v(i, 0) = r(i)/norm_residual;
  
  real rho  = norm_residual;

  vec_g = 0.0;
  vec_g(0) = rho;

  int k,k_end;
  for (k=0;k<k_max;k++){
    
    //cout << "Starting GMRES iteration number " << k+1 << endl;
    
    // Compute Krylov vector AP^(-1) v(k) 
    // 2 steps: First solve Px = v, then apply A to x.

    // if P=I => apply A directly: 
    for (int i=0;i<A.size(0);i++) vec_tmp1(i) = mat_v(i,k);
    A.mult(vec_tmp1,vec_tmp2);
    for (int i = 0; i< A.size(0); i++)
		mat_v(i, k+1) = vec_tmp2(i);

    for (int j=0;j<k+1;j++){
      mat_H(j,k) = 0.0;
      for (int i=0;i<n;i++) mat_H(j,k) += mat_v(i,k+1) * mat_v(i,j);
      for (int i=0;i<n;i++) mat_v(i,k+1) -= mat_H(j,k) * mat_v(i,j);
    }
    
    norm_v = 0.0;
    for (int i=0;i<n;i++) norm_v += sqr(mat_v(i,k+1));
    norm_v = sqrt(norm_v);
    
    mat_H(k+1,k) = norm_v;
    
    // FIXME: Test for non orthogonality and reorthogonalise if necessary
    // if ( !TestForOrthogonality(mat_v) ){ 
    if ( true ){
      for (int j=0; j < k+1; j++){
	for (int i=0;i<n;i++) htmp = mat_v(i,k+1) * mat_v(i,j);
	mat_H(j,k) += htmp;
	for (int i=0;i<n;i++) mat_v(i,k+1) -= htmp*mat_v(i,j);
      }	  
    }
    
    for (int i=0;i<n;i++) mat_v(i,k+1) /= norm_v;
	 
    if ( k > 0 ){
      for (int j=0;j<k;j++){
		  tmp1 = mat_H(j,k);
		  tmp2 = mat_H(j+1,k);
		  mat_H(j,k) = vec_c(j)*tmp1 - vec_s(j)*tmp2;
		  mat_H(j+1,k) = vec_s(j)*tmp1 + vec_c(j)*tmp2;
      }
    }
    
    nu = sqrt( sqr(mat_H(k,k)) + sqr(mat_H(k+1,k)) );
    
    vec_c(k) = mat_H(k,k) / nu;
    vec_s(k) = - mat_H(k+1,k) / nu;
    
    mat_H(k,k) = vec_c(k)*mat_H(k,k) - vec_s(k)*mat_H(k+1,k);
    mat_H(k+1,k) = 0.0;
    
    tmp = vec_c(k)*vec_g(k) - vec_s(k)*vec_g(k+1);
    vec_g(k+1) = vec_s(k)*vec_g(k) + vec_c(k)*vec_g(k+1);
    vec_g(k) = tmp;
    
    rho = fabs(vec_g(k+1));
    
    //cout << "GMRES residual rho = " << rho << " rho/||b|| = " << rho/norm_b << endl;
	 
    if ( (rho < tol*norm_b) || (k == k_max-1) ){
      //display->Status(2,"GMRES iteration converged after %i iterations",k);
      k_end = k;
      no_iterations = k;
      break;
    }

  }
  
  k = k_end;
  
  //cout << "Postprocessing to obtain solution" << endl;
  for (int i=0;i<k+1;i++){ 
    vec_w(i)= vec_g(i);
    for (int j=0;j<k+1;j++){ 
      mat_r(i,j) = mat_H(i,j);
    }
  } 

  // Solve triangular system ry = w
  vec_y(k) = vec_w(k)/mat_r(k,k);
  for (int i=1;i<k+1;i++){
    vec_y(k-i) = vec_w(k-i);
    for (int j=0;j<i;j++) vec_y(k-i) -= mat_r(k-i,k-j)*vec_y(k-j);
    vec_y(k-i) /= mat_r(k-i,k-i);
  }
  
  tmpvec = 0.0;
  for (int i=0;i<n;i++){
    for (int j=0;j<k+1;j++) tmpvec(i) += mat_v(i,j)*vec_y(j);
  }

  /*
  if ( pc != NONE ){
    // Get solution from preconditioned problem (get u from Pu=x)
    solvePxv(A, tmpvec);
  }
  */

  for (int i=0;i<n;i++) x(i) += tmpvec(i);

  norm_residual = residual(A,x,b);

  //display->Status(2,"Residual = %1.4e, Residual/||b|| = %1.4e",norm_residual,norm_residual/norm_b);
  
  return norm_residual;
}
//-----------------------------------------------------------------------------
void KrylovSolver::solveCG(Matrix &A, Vector &x, Vector &b)
{
  // Only for symmetric, positive definit problems. 
  // Does not work for the standard way of applying 
  // dirichlet boundary conditions, since then the 
  // symmetry of the matrix is destroyed.

  cout << "Using conjugate gradient (CG) solver." << endl;
  
  int n = x.size();
  int k_max = 20;
  
  real norm_b = b.norm();
  
  // Compute start residual = b-Ax.
  Vector r(n);
  real norm_residual = residual(A,x,b);
  
  Vector rho(k_max+1);
  rho = 0.0;
  rho(0) = sqr(norm_residual);

  Vector w(n);
  w = 0.0;
  Vector p(n);
  p = 0.0;

  real alpha,beta,tmp;

  /*
  real diag = 0.0;
  for (int i=0;i<sz;i++){
    diag = A->GetDiagonal(i);
    if ( fabs(diag) > KW_EPS )
      A->ScaleRow(i,1.0/diag);
  }
  */

  for (int k=1;k<k_max;k++){
    
    cout << "Starting CG iteration number " << k << endl;
    
    if ( k==1 ){
      p = r;
    } else{
      beta = rho(k-1)/rho(k-2);
      for (int i=0;i<n;i++) p(i) = r(i) + beta*p(i);
    }      
    
    A.mult(p,w);
    
    tmp = p*w;
    alpha = rho(k-1)/tmp;
    
    x.add(alpha,p);
    
    r.add(-alpha,w);
    
    rho(k) = sqr(r.norm());
    
    cout << "CG residual rho = " << rho(k-1) << endl;
    
    if ( sqrt(rho(k-1)) < tol*norm_b ){
      cout << "CG iteration converged" << endl;
      break;
    }
    
  }

  norm_residual = residual(A,x,b);

  cout << "Residual/||b|| = " << norm_residual/norm_b << endl;
  cout << "Residual = " << norm_residual << endl;

}
//-----------------------------------------------------------------------------
/*
void KrylovSolver::applyMatrix(Matrix &A, real **x, int comp )
{
  // Preconditioner
  SISolver sisolver;

  switch( pc ){ 
  case RICHARDSON:
    sisolver.set(SISolver::RICHARDSON);
    break;
  case JACOBI:
    sisolver.set(SISolver::JACOBI);
    break;
  case GAUSS_SEIDEL:
    sisolver.set(SISolver::GAUSS_SEIDEL);
    break;
  case SOR:
    sisolver.set(SISolver::SOR);
    break;
  case NONE:
    break;
  default:
    cout << "SISolver::Solve(): Unknown preconditioner" << endl;
	 exit(1);
  }

  Vector tmp1(A.size(0)); 
  Vector tmp2(A.size(0)); 

  for (int i=0; i<A.size(0); i++)
	 tmp1(i) = x[i][comp];

  // Precondition 
  no_pc_sweeps = 1;
  if ( pc != NONE ){
    sisolver.set(no_pc_sweeps);
    sisolver.solve(A, tmp2, tmp1);
    tmp1 = tmp2;
  }      
    
  A.mult(tmp1,tmp2);

  for (int i = 0; i < A.size(0); i++)
	 x[i][comp+1] = tmp2(i);
}
//-----------------------------------------------------------------------------
void KrylovSolver::solvePxv(Matrix &A, Vector &x)
{
  // Preconditioner
  SISolver sisolver;
  
  switch ( pc ) { 
  case RICHARDSON:
    sisolver.set(SISolver::RICHARDSON);
    break;
  case JACOBI:
    sisolver.set(SISolver::JACOBI);
    break;
  case GAUSS_SEIDEL:
    sisolver.set(SISolver::GAUSS_SEIDEL);
    break;
  case SOR:
    sisolver.set(SOR);
    break;
  case NONE:
    break;
  default:
	 cout << "SISolver::Solve(): Unknown preconditioner" << endl;
	 exit(1);
  }

  Vector tmp1(x);

  // Solve preconditioned problem 
  sisolver.set(no_pc_sweeps);
  sisolver.solve(A, x, tmp1);
}
*/
//-----------------------------------------------------------------------------
void KrylovSolver::residual(Matrix &A, Vector &x, Vector &b, Vector &r)
{
  for (int i =0;i<A.size(0);i++)
    r(i) = b(i) - A.mult(x,i);
}
//-----------------------------------------------------------------------------
real KrylovSolver::residual(Matrix &A, Vector &x, Vector &b)
{
  real r = 0.0;
  
  for (int i=0;i<A.size(0);i++)
    r += sqr(b(i) - A.mult(x, i));

  return sqrt(r);
}
//-----------------------------------------------------------------------------
/*
bool KrylovSolver::TestForOrthogonality( real **v )
{
  // if (||Av_k||+delta*||v_k+1||=||Av_k||)  delta \approx 10^-3
  
  // FIXME: cout << "Reorthogonalization needed" << endl;
  
  return false;
}
*/
//-----------------------------------------------------------------------------
