// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Thomas Svedberg, 2004: BiCGSTAB solver

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/basic.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/SISolver.h>
#include <dolfin/KrylovSolver.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(Method method)
{
  this->method = method;
  pc  = NONE;
  tol = dolfin_get("krylov tolerance");
}
//-----------------------------------------------------------------------------
void KrylovSolver::setMethod(Method method)
{
  this->method = method;
}
//-----------------------------------------------------------------------------
void KrylovSolver::setPreconditioner(Preconditioner pc)
{
  this->pc = pc;
}
//-----------------------------------------------------------------------------
void KrylovSolver::solve(const Matrix& A, Vector& x, const Vector& b)
{
  if ( A.size(0) != A.size(1) ) {
    cout << "A = " << A << endl;
    dolfin_error("Matrix must be square.");
  }
  if ( A.size(0) != b.size() ) {
    cout << "A = " << A << endl;
    cout << "b = " << b << endl;
    dolfin_error("Incompatible matrix and vector dimensions.");
  }
  
  tol = dolfin_get("krylov tolerance");

  cout << "Using Krylov solver for linear system of " << b.size() << " unknowns." << endl;
  
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
  case BiCGSTAB:
    solveBiCGSTAB(A,x,b);
    break;
  default:
    dolfin_error("Unknown Krylov method.");
  }
}
//-----------------------------------------------------------------------------
void KrylovSolver::solveGMRES(const Matrix& A, Vector& x, const Vector& b)
{
  // Get parameters
  int max_no_restarts = dolfin_get("max no krylov restarts"); // max no restarts
  unsigned int k_max = dolfin_get("max no stored krylov vectors"); // no iterations before restart

  // Compute residual
  Vector r(b.size());
  real norm_residual = residual(A,x,b,r);

  // Restarted GMRES
  for (int i = 0; i < max_no_restarts; i++) {
    
    // Krylov iterations
    int no_iterations = restartedGMRES(A,x,b,r,k_max);

    // Evaluate residual
    norm_residual = residual(A,x,b,r);

    // Check residual
    if (norm_residual < (tol*b.norm())){
      dolfin_info("Restarted GMRES converged after %i iterations (residual %1.5e)",
		  i*k_max+no_iterations,norm_residual);
      break;
    }
    if (i == max_no_restarts-1)
      dolfin_error2("Restarted GMRES did not converge (%i iterations, residual %1.5e)",
		    i*k_max+no_iterations,norm_residual);

  }
}
//-----------------------------------------------------------------------------
int KrylovSolver::restartedGMRES(const Matrix& A, Vector& x, const Vector& b,
				 Vector& r, unsigned int k_max)
{
  //Flexible GMRES with right preconditioner.

  // Number of unknowns
  unsigned int n = x.size();
  
  // Initialization of temporary variables 
  Matrix mat_h(k_max+1,k_max+1, Matrix::dense);
  Matrix mat_v(1,n, Matrix::dense);
  Matrix mat_z(1,n, Matrix::dense);

  Vector vec_s(k_max+1);  
  Vector vec_c(k_max+1); 
  Vector vec_g(k_max+1); 
  Vector vec_z(n);
  Vector vec_v(n);

  real tmp1, tmp2, nu;
  real htmp = 0.0;
  
  // Compute start residual = b - Ax.
  real norm_residual = residual(A,x,b,r);

  // Set start values for v. 
  for (unsigned int i = 0; i < n; i++)
    mat_v[0][i] = vec_v(i) = r(i)/norm_residual;

  if (pc != NONE){
    no_pc_sweeps = dolfin_get("pc iterations"); // no pc iterations (sweeps
    solvePxu(A,vec_z,vec_v);
    A.mult(vec_z,vec_v);
    mat_z(0,all) = vec_z;
  }
  
  vec_g = 0.0;
  vec_g(0) = norm_residual;

  unsigned int k = 0;
  unsigned int k_end = 0;
  
  for (k = 0; k < k_max; k++) {

    // Use the modified Gram-Schmidt process with reorthogonalization to
    // find orthogonal Krylov vectors.
    
    if (k>0) mat_v.addrow(vec_v);

    if (pc != NONE){
      solvePxu(A,vec_z,vec_v);
      A.mult(vec_z,vec_v);
      if (k>0) mat_z.addrow(vec_z);
    }
    else{
      vec_z = vec_v;
      A.mult(vec_z,vec_v);
    }

    // Modified Gram-Schmit orthogonalization. 
    for (unsigned int j = 0; j < k+1; j++) {
      /* konstruktionen här är långsammare än den efterföljande
      htmp = vec_v * mat_v(j,all); 
      mat_h(j,k) = htmp;
      vec_v.add(-htmp,mat_v(j,all));
      */
      htmp = 0.0;
      for (unsigned int i = 0; i < n; i++) htmp += vec_v(i) * mat_v[j][i];
      mat_h[j][k] = htmp;
      for (unsigned int i = 0; i < n; i++) vec_v(i) -= htmp * mat_v[j][i];
    }
    mat_h[k+1][k] = vec_v.norm();
    
    // Test for non orthogonality and reorthogonalise if necessary.
    if ( reorthog(A,mat_v,vec_v,k) ){ 
      for (unsigned int j = 0; j < k+1; j++) {
	/* konstruktionen här är långsammare än den efterföljande
	htmp = vec_v * mat_v(j,all);
	mat_h(j,k) += htmp;
	vec_v.add(-htmp,mat_v(j,all));
	*/
	for (unsigned int i = 0; i < n; i++) htmp = vec_v(i)*mat_v[j][i];
	mat_h(j,k) += htmp;
	for (unsigned int i = 0; i < n; i++) vec_v(i) -= htmp*mat_v[j][i];
      }	  
      mat_h[k+1][k] = vec_v.norm();
    }
    
    // Normalize
    vec_v *= 1.0/mat_h[k+1][k]; 
	 
    // If k > 0, solve least squares problem using QR factorization
    // and Givens rotations.  
    if (k>0) {
      for (unsigned int j = 0; j < k; j++) {
	tmp1 = mat_h[j][k];
	tmp2 = mat_h[j+1][k];
	mat_h[j][k]   = vec_c(j)*tmp1 - vec_s(j)*tmp2;
	mat_h[j+1][k] = vec_s(j)*tmp1 + vec_c(j)*tmp2; 
      }
    }
    
    nu = sqrt( sqr(mat_h[k][k]) + sqr(mat_h[k+1][k]) );
    
    vec_c(k) =   mat_h[k][k]   / nu;
    vec_s(k) = - mat_h[k+1][k] / nu;
    
    mat_h[k][k] = vec_c(k)*mat_h[k][k] - vec_s(k)*mat_h[k+1][k];
    mat_h[k+1][k] = 0.0;
    
    tmp1 = vec_c(k)*vec_g(k) - vec_s(k)*vec_g(k+1);
    vec_g(k+1) = vec_s(k)*vec_g(k) + vec_c(k)*vec_g(k+1);
    vec_g(k) = tmp1;
    
    // If residual rho = Ax-b less than tolerance 
    // (normalized with ||b||) we are done.
    if ( (fabs(vec_g(k+1)) < tol*b.norm()) || (k==k_max-1) ){
      k_end = k;
      break;
    }
  }
  k = k_end;
 
  // Postprocess to obtain solution by solving system Ry=w.
  Matrix mat_r(k_max+1,k_max+1, Matrix::dense);
  mat_r = mat_h;
  Vector vec_w(vec_g);
  Vector vec_y(k_end+1); 

  // solve triangular system ry = w
  vec_y(k) = vec_w(k)/mat_r[k][k];
  for (unsigned int i = 1; i < k+1; i++) {
    vec_y(k-i) = vec_w(k-i);
    for (unsigned int j = 0; j < i; j++) vec_y(k-i) -= mat_r[k-i][k-j]*vec_y(k-j);
    vec_y(k-i) /= mat_r[k-i][k-i];
  }

  if ( pc != NONE){
    vec_z = 0.0;
    mat_z.multt(vec_y,vec_z);
    x += vec_z;
  }
  else{
    vec_v = 0.0;
    mat_v.multt(vec_y,vec_v);
    x += vec_v;
  }
  
  return k_end;
}
//-----------------------------------------------------------------------------
void KrylovSolver::solveCG(const Matrix &A, Vector &x, const Vector &b)
{
  // Only for symmetric, positive definite problems.

  unsigned int n = x.size();
  unsigned int k_max = dolfin_get("max no cg iterations"); // max no iterations
  
  Vector rho(k_max+1);
  
  // Compute start residual = b-Ax.
  Vector r(n), z;
  real norm_residual = residual(A,x,b,r);

  if (pc != NONE){ 
    no_pc_sweeps = dolfin_get("pc iterations"); // no pc iterations (sweeps)
    solvePxu(A,z,r);
    rho(0) = r*z;
  }
  else rho(0) = sqr(norm_residual);

  Vector w(n);
  Vector p(n);

  real alpha,beta,tmp;

  unsigned int k;
  for (k = 1; k < k_max; k++) {

    if ( k==1 ) {
      p = (pc!=NONE) ? z : r;
    } 
    else{
      beta = rho(k-1)/rho(k-2);
      if (pc != NONE){
	for (unsigned int i = 0; i < n; i++) p(i) = z(i) + beta*p(i);
      }
      else{
	for (unsigned int i = 0; i < n; i++) p(i) = r(i) + beta*p(i);
      }
    }      
    
    A.mult(p,w);
    
    alpha = rho(k-1)/ (p*w);
    
    x.add(alpha,p);
    r.add(-alpha,w);
    
    if (pc != NONE){
      z = 0.0;
      solvePxu(A,z,r);
      rho(k) = r*z;
    }
    else{
      rho(k) = sqr(r.norm());
    }

    if (k == k_max-1){
      norm_residual = residual(A,x,b,r);
      dolfin_info("CG iterations did not converge: residual = %1.5e",
		  norm_residual);
      exit(1);
    }
    
    tmp = (pc!=NONE) ? r.norm(1) : sqrt(rho(k));
    if (tmp  < tol*b.norm() ) break;
  }

  norm_residual = residual(A,x,b,r);
  dolfin_info("CG converged after %i iterations (residual = %1.5e)."
	      ,k, norm_residual);
}
//-----------------------------------------------------------------------------
void KrylovSolver::solveBiCGSTAB(const Matrix &A, Vector &x, const Vector &b)
{
  size_t n (x.size());
  real rho_1 = 0.0;
  real rho_2 = 0.0;
  real alpha = 0.0;
  real beta  = 0.0;
  real omega = 0.0;
  real b_norm(b.norm());
  
  Vector r(n);
  real norm_residual = residual (A, x, b, r);
  
  if (norm_residual  < tol * b_norm) return;
  
  Vector rtilde (r);
  
  Vector p(n), phat(n), s(n), shat(n), t(n), v(n);
  
  size_t iter (0);
  while (1) {
    
    rho_1 = rtilde * r;
    
    if (fabs (rho_1) < 1e-25) {
      dolfin_info("BICGSTAB breakdown: rho_1 = %g", rho_1);
      exit (1);
    }
    
    if (iter == 0)
      p = r;
    else {
      if (fabs (omega) < 1e-25) {
	dolfin_info("BICGSTAB breakdown: omega = %g", omega);
	exit (1);
      }
      
      beta = (rho_1 * alpha) / (rho_2 * omega);
      
      for (size_t i = 0; i < n; i++)
	p(i) = r(i) + beta * (p(i) - omega * v(i));
    }
    if (pc != NONE) {
      phat = 0.0;
      solvePxu (A, phat, p);
    } else
      phat = p;
    
    A.mult (phat, v);
    
    alpha = rho_1 / (v * rtilde);
    
    for (size_t i = 0; i < n; i++)
      s(i) = r(i) - alpha * v(i);
    
    if (s.norm() < tol * b_norm) {
      for (size_t i = 0; i < n; i++)
	x(i) += alpha * phat(i);
      dolfin_info("BicCGSTAB converged after %i iterations (residual %1.5e)",
		  iter, s.norm());
      break;
    }
    
    if (pc != NONE) {
      shat = 0.0;
      solvePxu (A, shat, s);
    } else
      shat = s;
    
    A.mult (shat, t);
    
    omega = (t * s) / (t * t);
    
    for (size_t i = 0; i < n; i++) {
      x(i) += alpha * phat(i) + omega * shat(i);
      r(i) = s(i) - omega * t(i);
    }
    
    rho_2 = rho_1;
    
    ++iter;
  }
}
//-----------------------------------------------------------------------------
void KrylovSolver::solvePxu(const Matrix &A, Vector &x, Vector &u)
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
//-----------------------------------------------------------------------
real KrylovSolver::residual(const Matrix& A, Vector& x, const Vector& b, 
			    Vector& r)
{
  int n = b.size();

  r.init(n);
  A.mult(x,r);

  real sum = 0.0;
  for (int i = 0; i < n; i++) {
    r(i) = b(i) - r(i);
    sum += r(i)*r(i);
  }
  
  return sqrt(sum);
}
//-----------------------------------------------------------------------------
bool KrylovSolver::reorthog(const Matrix& A, Matrix& v, Vector &x, int k)
{
  // reorthogonalize if ||Av_k||+delta*||v_k+1||=||Av_k|| to working precision 
  // with  delta \approx 10^-3
  Vector Av(v.size(1));
  Vector w(v.size(1));
 
  w = v(k,all);

  A.mult(w,Av);
  
  real delta = 1.0e-3;
  real norm_Av = Av.norm();

  return ((norm_Av + delta*x.norm()) - norm_Av) < DOLFIN_EPS;
}
//-----------------------------------------------------------------------------
