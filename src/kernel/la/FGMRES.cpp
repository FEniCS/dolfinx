// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/FGMRES.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FGMRES::FGMRES(const Matrix& A, unsigned int restarts, unsigned int maxiter, 
	       real tol, Preconditioner& pc)
  : Preconditioner(), A(A), pc(pc)
{
  this -> restarts = restarts;
  this -> maxiter = maxiter;
  this -> tol = tol;
  solve2convergence = true;
}
//-----------------------------------------------------------------------------
FGMRES::FGMRES(const Matrix& A, unsigned int maxiter, 
	   real tol, Preconditioner& pc)
  : Preconditioner(), A(A), pc(pc)
{
  restarts = 1;
  this -> maxiter = maxiter;
  this -> tol = tol;
  solve2convergence = false;
}
//-----------------------------------------------------------------------------
FGMRES::~FGMRES()
{
 
}
//-----------------------------------------------------------------------------
void FGMRES::solve(const Matrix& A, Vector& x, const Vector& b,
		      unsigned int restarts, unsigned int maxiter, real tol,
		      Preconditioner& pc)
{
  // Create a FGMRES object
  FGMRES fgmres(A, restarts, maxiter, tol, pc);
  
  // Solve linear system
  fgmres.solve(x, b);
}
//-----------------------------------------------------------------------------
void FGMRES::solve(Vector& x, const Vector& b)
{
  // Check compatibility in the matrix and vectors.
  check(A, x, b);

  bnorm = b.norm();
  // Check if b=0 => x=0.
  if ( bnorm < DOLFIN_EPS ){
    x = 0.0;
    return;
  }
  
  // Compute residual
  Vector r(b.size());
  real rnorm = residual(A,x,b,r);

  // Restarted GMRES
  for ( unsigned int i = 0; i < restarts; i++ ){
    
    // Krylov iterations
    unsigned int noiter = iterator(x,b,r);
    
    // Check stopping criterion
    if ( solve2convergence ) { 
      
      // Evaluate residual
      rnorm = residual(A,x,b,r);

      // Check residual
      if ( rnorm < tol*bnorm ){
	dolfin_info("Restarted FGMRES converged after %i iterations (residual %1.5e)",
		    i*maxiter+noiter, rnorm);
	break;
      }
      if ( i == restarts-1 )
	dolfin_error2("Restarted FGMRES did not converge (%i iterations, residual %1.5e)",
		     i*maxiter+noiter, rnorm);
    }  
  }
}
//-----------------------------------------------------------------------------
unsigned int FGMRES::iterator(Vector& x, const Vector& b, Vector& r)
{
  // Number of unknowns
  unsigned int n = x.size();
  
  // Initialization of temporary variables 
  Matrix mat_h(maxiter+1,maxiter+1, Matrix::dense);
  Matrix mat_v(1,n, Matrix::dense);
  Matrix mat_z(1,n, Matrix::dense);

  Vector vec_s(maxiter+1);  
  Vector vec_c(maxiter+1); 
  Vector vec_g(maxiter+1); 
  Vector vec_z(n);
  Vector vec_v(n);

  real tmp1, tmp2, nu;
  real htmp = 0.0;
  
  // Compute start residual = b - Ax.
  real rnorm = residual(A,x,b,r);

  // Set start values for v. 
  for (unsigned int i = 0; i < n; i++)
    mat_v[0][i] = vec_v(i) = r(i)/rnorm;
  
  vec_g = 0.0;
  vec_g(0) = rnorm;

  unsigned int k = 0;
  unsigned int k_end = 0;
  
  for (k = 0; k < maxiter; k++) {

    // Use the modified Gram-Schmidt process with reorthogonalization to
    // find orthogonal Krylov vectors.
    if (k != 0) mat_v.addrow(vec_v);
    
    pc.solve(vec_z,vec_v);
    A.mult(vec_z,vec_v);
         
    if (k == 0) 
      for (unsigned int i=0;i<n;i++) mat_z[0][i] = vec_z(i); 
    else
      mat_z.addrow(vec_z);
    
    // Modified Gram-Schmit orthogonalization. 
    for (unsigned int j = 0; j < k+1; j++) {
      htmp = 0.0;
      for (unsigned int i = 0; i < n; i++) htmp += vec_v(i) * mat_v[j][i];
      mat_h[j][k] = htmp;
      for (unsigned int i = 0; i < n; i++) vec_v(i) -= htmp * mat_v[j][i];
    }
    mat_h[k+1][k] = vec_v.norm();
    
    // Test for non orthogonality and reorthogonalise if necessary.
    if ( reorthog(mat_v,vec_v,k) ){ 
      for (unsigned int j = 0; j < k+1; j++) {
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
    
    // If residual rho = Ax-b less than tolerance (normalized with ||b||)
    // we are done.  

    if ( (fabs(vec_g(k+1)) < tol*bnorm) || (k == maxiter-1) ){
      k_end = k;
      break;
    }
  }
  k = k_end;

  // Postprocess to obtain solution by solving system Ry=w.
  Vector vec_w(vec_g);
  Vector vec_y(k_end+1); 
  
  // solve triangular system ry = w
  vec_y(k) = vec_w(k)/mat_h[k][k];
  for (unsigned int i = 1; i < k+1; i++) {
    vec_y(k-i) = vec_w(k-i);
    for (unsigned int j = 0; j < i; j++) vec_y(k-i) -= mat_h[k-i][k-j]*vec_y(k-j);
    vec_y(k-i) /= mat_h[k-i][k-i];
  }

  vec_z = 0.0;
  mat_z.multt(vec_y,vec_z);
  x += vec_z;
  
  return k_end;
}
//-----------------------------------------------------------------------------
bool FGMRES::reorthog(Matrix& v, Vector &x, int k)
{
  // reorthogonalize if ||Av_k||+delta*||v_k+1||=||Av_k|| to working precision 
  // with  delta \approx 10^-3
  Vector Av(v.size(1));
  Vector w(v.size(1));
 
  w = v(k,all);

  A.mult(w,Av);
  
  real delta = 1.0e-3;
  real Avnorm = Av.norm();

  return ((Avnorm + delta*x.norm()) - Avnorm) < DOLFIN_EPS;
}
//-----------------------------------------------------------------------------
