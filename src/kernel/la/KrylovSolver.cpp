// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/basic.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/DenseMatrix.h>
#include <dolfin/SISolver.h>
#include <dolfin/Settings.h>
#include <dolfin/KrylovSolver.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(Method method)
{
  this->method = method;
  pc  = NONE;
  tol = 1.0e-10;
}
//-----------------------------------------------------------------------------
void KrylovSolver::setMethod(Method method)
{
  this->method = method;
}
//-----------------------------------------------------------------------------
void KrylovSolver::solve(Matrix &A, Vector &x, Vector &b)
{
  if ( A.size(0) != A.size(1) )
    dolfin_error("Matrix must be square.");
  if ( A.size(0) != b.size() )
    dolfin_error("Incompatible matrix and vector dimensions.");
  
  cout << "Using Krylov solver for linear system of " << b.size() << " unknowns" << endl;
  
  // Check if we need to resize x
  if (x.size() != b.size()) x.init(b.size());

  // Check if b=0 => x=0
  if (b.norm() < DOLFIN_EPS){
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
    dolfin_error("Unknown Krylov method.");
  }
}
//-----------------------------------------------------------------------------
void KrylovSolver::solveGMRES(Matrix &A, Vector &x, Vector &b)
{
  int k_max,max_no_restarts;
  Settings::get("max no krylov restarts", &max_no_restarts); // max no restarts
  Settings::get("max no stored krylov vectors", &k_max); // no iterations before restart

  int no_iterations;

  real norm_residual = getResidual(A,x,b);
  for (int i=0;i<max_no_restarts;i++){
    no_iterations = restartedGMRES(A,x,b,k_max);
    norm_residual = getResidual(A,x,b);
    if (norm_residual < (tol*b.norm())){
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
int KrylovSolver::restartedGMRES(Matrix &A, Vector &x, Vector &b, int k_max)
{
  // Solve preconditioned problem Ax = AP^(-1)Px = b, 
  // by first solving AP^(-1)u=b through GMRES, then 
  // solve Px = u to get x. Extrem cases is P = A (changes nothing) 
  // and P = I (corresponds to no preconditioner). 
  // AP^(-1)u=b is solved to a tolerance rho/|b| < tol, 
  // where rho is the norm of the residual b-AP^(-1)u = b-Ax,
  // at a maximal number of iterations k_max, 
  // starting from startvector u = Px. 

  // number of unknowns
  int n = x.size();
  
  // Initializing unknown u=Px
  Vector u(x);
  applyPxu(A,x,u);

  // initialization of temporary variables 
  DenseMatrix mat_h(k_max+1,k_max+1);
  DenseMatrix mat_r(k_max+1,k_max+1);
  DenseMatrix mat_v(n,k_max+1);

  Vector vec_s(k_max+1); 
  Vector vec_y(k_max+1); 
  Vector vec_w(k_max+1); 
  Vector vec_c(k_max+1); 
  Vector vec_g(k_max+1); 

  Vector vec_tmp1(n);
  Vector vec_tmp2(n);

  real tmp1,tmp2,nu,norm_v,htmp;

  // norm of rhs b
  real norm_b = b.norm();
  
  // Compute start residual = b-AP^(-1)u = b-Ax.
  Vector r(n);
  real norm_residual = getResidual(A,x,b,r);

  // Set start values for v,rho,beta,vec_g 
  for (int i=0;i<n;i++) mat_v(i,0) = r(i)/norm_residual;

  // set start column of v to residual/|residual|
  for (int i = 0; i <n; i++)
    mat_v(i, 0) = r(i)/norm_residual;

  real rho  = norm_residual;

  vec_g = 0.0;
  vec_g(0) = rho;

  int k,k_end;
  for (k=0;k<k_max;k++){
    // Use the modified Gram-Schmidt process find orthogonal Krylov vectors
    //
    // The computation of the Krylov vector AP^(-1) v(k) includes  
    // 2 steps: First solve Pw = v, then apply A to w.
    solvePxv(A,vec_tmp1,mat_v,k);
    A.mult(vec_tmp1,vec_tmp2);
    for (int i=0;i<n;i++) mat_v(i,k+1) = vec_tmp2(i);

    for (int j=0;j<k+1;j++){
      mat_h(j,k) = 0.0;
      for (int i=0;i<n;i++) mat_h(j,k) += mat_v(i,k+1) * mat_v(i,j);
      for (int i=0;i<n;i++) mat_v(i,k+1) -= mat_h(j,k) * mat_v(i,j);
    }
    norm_v = 0.0;
    for (int i=0;i<n;i++) norm_v += sqr(mat_v(i,k+1));
    norm_v = sqrt(norm_v);
    mat_h(k+1,k) = norm_v;
    
    // Test for non orthogonality and reorthogonalise if necessary
    if ( reOrthogonalize(A,mat_v,k) ){ 
      for (int j=0;j<k+1;j++){
	for (int i=0;i<n;i++) htmp = mat_v(i,k+1)*mat_v(i,j);
	mat_h(j,k) += htmp;
	for (int i=0;i<n;i++) mat_v(i,k+1) -= htmp*mat_v(i,j);
      }	  
      norm_v = 0.0;
      for (int i=0;i<n;i++) norm_v += sqr(mat_v(i,k+1));
      norm_v = sqrt(norm_v);
      mat_h(k+1,k) = norm_v;
    }
    
    for (int i=0;i<n;i++) mat_v(i,k+1) /= norm_v;
	 
    // If k > 0, solve least squares problem using QR factorization and Givens rotations  
    if ( k > 0 ){
      for (int j=0;j<k;j++){
	tmp1 = mat_h(j,k);
	tmp2 = mat_h(j+1,k);
	mat_h(j,k)   = vec_c(j)*tmp1 - vec_s(j)*tmp2;
	mat_h(j+1,k) = vec_s(j)*tmp1 + vec_c(j)*tmp2;
      }
    }
    
    nu = sqrt( sqr(mat_h(k,k)) + sqr(mat_h(k+1,k)) );
    
    vec_c(k) =   mat_h(k,k)   / nu;
    vec_s(k) = - mat_h(k+1,k) / nu;
    
    mat_h(k,k) = vec_c(k)*mat_h(k,k) - vec_s(k)*mat_h(k+1,k);
    mat_h(k+1,k) = 0.0;
    
    tmp1 = vec_c(k)*vec_g(k) - vec_s(k)*vec_g(k+1);
    vec_g(k+1) = vec_s(k)*vec_g(k) + vec_c(k)*vec_g(k+1);
    vec_g(k) = tmp1;
    
    rho = fabs(vec_g(k+1));
    
    // if residual rho = Ax-b less than tolerance (normalized with ||b||) we are done
    if ( (rho < tol*norm_b) || (k == k_max-1) ){
      k_end = k;
      break;
    }

  }
  k = k_end;
  
  // postprocess to obtain solution by solving system Ry=w
  for (int i=0;i<k+1;i++){ 
    vec_w(i)= vec_g(i);
    for (int j=0;j<k+1;j++){ 
      mat_r(i,j) = mat_h(i,j);
    }
  } 
  // solve triangular system ry = w
  vec_y(k) = vec_w(k)/mat_r(k,k);
  for (int i=1;i<k+1;i++){
    vec_y(k-i) = vec_w(k-i);
    for (int j=0;j<i;j++) vec_y(k-i) -= mat_r(k-i,k-j)*vec_y(k-j);
    vec_y(k-i) /= mat_r(k-i,k-i);
  }
  
  vec_tmp1 = 0.0;
  for (int i=0;i<n;i++){
    for (int j=0;j<k+1;j++) vec_tmp1(i) += mat_v(i,j)*vec_y(j);
  }

  for (int i=0;i<n;i++) u(i) += vec_tmp1(i);

  // Get solution from preconditioned problem (get x from Px=u)
  solvePxu(A,x,u);

  return k_end;
}
//-----------------------------------------------------------------------------
void KrylovSolver::solveCG(Matrix &A, Vector &x, Vector &b)
{
  // Only for symmetric, positive definite problems. 
  // Does not work for the standard way of applying 
  // Dirichlet boundary conditions, since then the 
  // symmetry of the matrix is destroyed.

  int n = x.size();
  int k_max;
  Settings::get("max no cg iterations", &k_max); // max no iterations
  
  real norm_b = b.norm();
  
  // Compute start residual = b-Ax.
  Vector r(n);
  real norm_residual = getResidual(A,x,b,r);
  
  Vector rho(k_max+1);
  rho = 0.0;
  rho(0) = sqr(norm_residual);

  Vector w(n);
  Vector p(n);

  real alpha,beta,tmp;

  int k;
  for (k=1;k<k_max;k++){

    if (k == 1){
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

    if (k == k_max-1){
      norm_residual = getResidual(A,x,b);
      cout << "CG iterations did not converge: residual = " << norm_residual << endl;
      exit(1);
    }

    if ( sqrt(rho(k-1)) < tol*norm_b ) break;
  }

  norm_residual = getResidual(A,x,b);
  cout << "CG converged after " << k << " iterations" 
       << " (residual = " << norm_residual << ")." << endl;
}
//-----------------------------------------------------------------------------
void KrylovSolver::applyPxu(Matrix &A, Vector &x, Vector &u)
{
  // Solve preconditioned problem Px=u
  SISolver sisolver;
  
  switch ( pc ) { 
  case RICHARDSON:
    sisolver.setMethod(SISolver::RICHARDSON);
    break;
  case JACOBI:
    sisolver.setMethod(SISolver::JACOBI);
    break;
  case GAUSS_SEIDEL:
    sisolver.setMethod(SISolver::GAUSS_SEIDEL);
    break;
  case SOR:
    sisolver.setMethod(SISolver::SOR);
    break;
  case NONE:
    u = x;
    return;
    break;
  default:
	 dolfin_error("Unknown preconditioner.");
  }

  sisolver.setNoSweeps(no_pc_sweeps);
  sisolver.solve(A,x,u);
}
//-----------------------------------------------------------------------------
void KrylovSolver::solvePxu(Matrix &A, Vector &x, Vector &u)
{
  // Solve preconditioned problem Px=u
  SISolver sisolver;
  
  switch ( pc ) { 
  case RICHARDSON:
    sisolver.setMethod(SISolver::RICHARDSON);
    break;
  case JACOBI:
    sisolver.setMethod(SISolver::JACOBI);
    break;
  case GAUSS_SEIDEL:
    sisolver.setMethod(SISolver::GAUSS_SEIDEL);
    break;
  case SOR:
    sisolver.setMethod(SISolver::SOR);
    break;
  case NONE:
    x = u;
    return;
    break;
  default:
    dolfin_error("Unknown preconditioner.");
  }

  sisolver.setNoSweeps(no_pc_sweeps);
  sisolver.solve(A,x,u);
}
//-----------------------------------------------------------------------------
void KrylovSolver::solvePxv(Matrix &A, Vector &x, DenseMatrix &v, int k)
{
  // Solve preconditioned problem Px=v
  SISolver sisolver;
  
  switch ( pc ) { 
  case RICHARDSON:
    sisolver.setMethod(SISolver::RICHARDSON);
    break;
  case JACOBI:
    sisolver.setMethod(SISolver::JACOBI);
    break;
  case GAUSS_SEIDEL:
    sisolver.setMethod(SISolver::GAUSS_SEIDEL);
    break;
  case SOR:
    sisolver.setMethod(SISolver::SOR);
    break;
  case NONE:
    for (int i=0;i<v.size(0);i++) x(i) = v(i,k);
    return;
    break;
  default:
    dolfin_error("Unknown preconditioner.");
  }

  Vector tmp(v.size(0));
  for (int i=0;i<v.size(0);i++) tmp(i) = v(i,k);
  sisolver.setNoSweeps(no_pc_sweeps);
  sisolver.solve(A,x,tmp);
}
//-----------------------------------------------------------------------------
real KrylovSolver::getResidual(Matrix &A, Vector &x, Vector &b, Vector &r)
{
  r.init(x.size());
  for (int i =0;i<A.size(0);i++) 
    r(i) = b(i) - A.mult(x,i);

  return r.norm();
}
//-----------------------------------------------------------------------------
real KrylovSolver::getResidual(Matrix &A, Vector &x, Vector &b)
{
  real norm_r = 0.0;
  for (int i=0;i<A.size(0);i++)
    norm_r += sqr(b(i) - A.mult(x,i));

  return sqrt(norm_r);
}
//-----------------------------------------------------------------------------
bool KrylovSolver::reOrthogonalize(Matrix &A, DenseMatrix &v, int k)
{
  // reorthogonalize if ||Av_k||+delta*||v_k+1||=||Av_k|| to working precision 
  // with  delta \approx 10^-3
  
  // The computation of the Krylov vector AP^(-1) v(k) includes  
  // 2 steps: First solve Pw = v, then apply A to w.
  Vector Av(v.size(0));
  Vector w(v.size(0));
  solvePxv(A,w,v,k);
  A.mult(w,Av);

  real norm_v = 0.0;
  for (int i=0;i<v.size(0);i++) norm_v += sqr(v(i,k+1));
  norm_v = sqrt(norm_v);
  
  real delta = 1.0e-3;

  if ( ((Av.norm() + delta*norm_v) - Av.norm()) < DOLFIN_EPS ) return true; 
  else return false;
}
//-----------------------------------------------------------------------------
