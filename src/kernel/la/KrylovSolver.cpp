// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Display.hh>
#include "KrylovSolver.hh"
#include "SISolver.hh"
#include <unistd.h>
#include "utils.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver()
{
  A = 0;

  krylov_method = gmres;
  pc            = pc_none;
  tol           = 1.0e-10;

  mat_H = 0;
  mat_r = 0;
  mat_v = 0;
  
  vec_s = 0;
  vec_y = 0;
  vec_w = 0;
  vec_c = 0;
  vec_g = 0;
  
}
//-----------------------------------------------------------------------------
void KrylovSolver::SetMethod(KrylovMethod krylov_method)
{
  this->krylov_method = krylov_method;
}
//-----------------------------------------------------------------------------
void KrylovSolver::Solve(SparseMatrix *A, Vector *x, Vector *b)
{
  SetMatrix(A);
  Solve(x,b);
}
//-----------------------------------------------------------------------------
void KrylovSolver::Solve(Vector* x, Vector* b)
{
  if (x->size()!=b->size())
    x->resize(b->size());
  
  norm_b = b->norm();
  if ( norm_b == 0 ){
    (*x) = 0.0;
    return;
  }

  switch( krylov_method ){ 
  case gmres:
    SolveGMRES(x,b);
    break;
  case cg:
    SolveCG(x,b);
    break;
  default:
    display->InternalError("KrylovSolver::Solve()","Krylov method not implemented");
  }
}
//-----------------------------------------------------------------------------
void KrylovSolver::SolveGMRES(Vector* x, Vector* b)
{
  // FIXME: Should be parameters
  int k_max             = 20;
  int max_no_iterations = 100;

  int n = x->size();

  // Allocate memory for arrays
  AllocateArrays(n,k_max);
  
  norm_residual = GetResidual(x,b);
  
  for (int i=0;i<max_no_iterations;i++){

    SolveGMRES_restart_k(x,b,k_max);
	 norm_residual = GetResidual(x,b);

	 if ( norm_residual < (tol*norm_b) ){
      if ( i > 0 )
		  display->Status(2,"Restarted GMRES converged after %i iterations (restarted after %i)",
								i*k_max+no_iterations,k_max);
		else
		  display->Status(2,"GMRES converged after %i iterations",no_iterations);

      display->Status(2,"Residual = %1.4e, Residual/||b|| = %1.4e",norm_residual,norm_residual/norm_b);
      break;
    }
	 
    if (i == max_no_iterations-1) 
      display->Error("GMRES iterations did not converge, ||res|| = %1.4e",norm_residual);
  }

  // Delete temporary memory
  DeleteArrays(n,k_max);
  
}
//-----------------------------------------------------------------------------
real KrylovSolver::SolveGMRES_restart_k(Vector* x, Vector* b, int k_max)
{
  // Solve preconditioned problem Ax = AP^(-1)Px = b, 
  // by first solving AP^(-1)v=b through GMRES, then 
  // solve Px = v to get x. Extrem cases is P = A (changes nothing) 
  // and P = I (corresponds to no preconditioner). 
  // AP^(-1)v=b is solved to a tolerance rho/|b| < tol, 
  // where rho is the norm of the residual b-AP^(-1)v = b-Ax,
  // at a maximal number of iterations kmax, 
  // starting from startvector v = Px. 

  int n = x->size();

  real norm_b = b->norm();
  
  // Compute start residual = b-AP^(-1)u = b-Ax.
  Vector residual(n);
  ComputeResidual(x,b,&residual);
  norm_residual = residual.norm();

  for (int i=0;i<n;i++) mat_v[i][0] = residual(i)/norm_residual;
      
  Vector tmpvec(n);

  real rho  = norm_residual;
  real beta = rho;

  real tmp1,tmp2;
  
  for (int i=0; i < k_max+1; i++) vec_g[i] = 0.0;
  vec_g[0] = rho;

  real tmp,nu,norm_v,htmp;
  
  int k,k_end;
  for (k=0; k < k_max; k++){
    
    display->Message(5,"Start GMRES iteration number %i",k+1);
    
    // Compute Krylov vector AP^(-1) v(k) 
    // 2 steps: First solve Px = v, then apply A to x.
    ApplyMatrix(mat_v,k);
    
    for (int j=0; j < k+1; j++){
      mat_H[j][k] = 0.0;
      for (int i=0;i<n;i++) mat_H[j][k]   += mat_v[i][k+1]*mat_v[i][j];
      for (int i=0;i<n;i++) mat_v[i][k+1] -= mat_H[j][k]*mat_v[i][j];
    }
    
    norm_v = 0.0;
    for (int i=0;i<n;i++) norm_v += sqr(mat_v[i][k+1]);
    norm_v = sqrt(norm_v);
    
    mat_H[k+1][k] = norm_v;
    
    // Test for non orthogonality and reorthogonalise if necessary
    if ( !TestForOrthogonality(mat_v) ){
      for (int j=0; j < k+1; j++){
		  for (int i=0;i<n;i++) htmp = mat_v[i][k+1]*mat_v[i][j];
		  mat_H[j][k] += htmp;
		  for (int i=0;i<n;i++) mat_v[i][k+1] -= htmp*mat_v[i][j];
      }	  
    }
    
    for (int i=0;i<n;i++) mat_v[i][k+1] /= norm_v;
	 
    if ( k > 0 ){
      for (int j=0; j < k; j++){
		  tmp1 = mat_H[j][k];
		  tmp2 = mat_H[j+1][k];
		  mat_H[j][k]   = vec_c[j]*tmp1 - vec_s[j]*tmp2;
		  mat_H[j+1][k] = vec_s[j]*tmp1 + vec_c[j]*tmp2;
      }
    }
    
    nu   = sqrt( sqr(mat_H[k][k]) + sqr(mat_H[k+1][k]) );
    
    vec_c[k] = mat_H[k][k] / nu;
    vec_s[k] = - mat_H[k+1][k] / nu;
    
    mat_H[k][k]   = vec_c[k]*mat_H[k][k] - vec_s[k]*mat_H[k+1][k];
    mat_H[k+1][k] = 0.0;
    
    tmp    = vec_c[k]*vec_g[k] - vec_s[k]*vec_g[k+1];
    vec_g[k+1] = vec_s[k]*vec_g[k] + vec_c[k]*vec_g[k+1];
    vec_g[k]   = tmp;
    
    rho = fabs(vec_g[k+1]);
    
    display->Message(5,"GMRES residual rho = %1.4e, rho/||b|| = %1.4e, tol = %1.4e",rho,rho/norm_b,tol);
	 
    if ( (rho < tol * norm_b) || (k == k_max-1) ){
      //display->Status(2,"GMRES iteration converged after %i iterations",k);
      k_end = k;
      no_iterations = k;
      break;
    }
  }
  
  k = k_end;
  
  display->Message(5,"Postprocess to obtain solution");
  
  for (int i=0; i < k+1; i++){ 
    vec_w[i] = vec_g[i];
    for (int j=0; j < k+1; j++){ 
      mat_r[i][j] = mat_H[i][j];
    }
  } 

  // Solve triangular system ry = w
  vec_y[k] = vec_w[k] / mat_r[k][k];
  for (int i=1; i < k+1; i++){
    vec_y[k-i] = vec_w[k-i];
    for (int j=0; j < i; j++) vec_y[k-i] -= mat_r[k-i][k-j]*vec_y[k-j];
    vec_y[k-i] /= mat_r[k-i][k-i];
  }
  
  tmpvec = 0.0;
  for (int i=0;i<n;i++){
    for (int j=0; j < k+1; j++) tmpvec(i) += mat_v[i][j]*vec_y[j];
  }

  if ( pc != pc_none ){
    // Get solution from preconditioned problem (get u from Pu=x)
    SolvePxv(&tmpvec);
  }

  for (int i=0;i<n;i++)
	 (*x)(i) += tmpvec(i);

  norm_residual = GetResidual(x,b);

  //display->Status(2,"Residual = %1.4e, Residual/||b|| = %1.4e",norm_residual,norm_residual/norm_b);
  
  return norm_residual;
}
//-----------------------------------------------------------------------------
void KrylovSolver::SolveCG(Vector* x, Vector* b)
{
  // Only for symmetric, positive definit problems. 
  // Does not work for the standard way of applying 
  // dirichlet boundary conditions, since then the 
  // symmetry of the matrix is destroyed.

  int sz = x->size();
  int k_max = 20;
  
  real norm_b = b->norm();
  
  // Compute start residual = b-Ax.
  Vector residual(sz);
  norm_residual = GetResidual(x,b);

  real *rho;
  rho = new real[k_max+1]; 
  for (int i = 0; i < k_max+1; i++) rho[i] = 0.0;
  rho[0] = sqr(norm_residual);

  Vector w(sz);
  w = 0.0;
  Vector p(sz);
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

  for (int k=1; k < k_max; k++){
    
    display->Message(0,"Start CG iteration number %i",k);
    
    display->Message(0,"rho(k-1) = %f, rho(k-2) = %f",rho[k-1],rho[k-2]);
    
    if ( k==1 ){
      p = residual;
    } else{
      beta = rho[k-1]/rho[k-2];
      for (int i=0; i < sz; i++)
		  p(i) = residual(i) + beta*p(i);
    }      
    
    ApplyMatrix(&p,&w);
    
    tmp = p * w;
    alpha = rho[k-1]/tmp;
    
    display->Message(0,"tmp = %f, alpha = %f, norm_p = %f, norm_w = %f",tmp,alpha,p.norm(),w.norm());
    
    x->add(alpha,p);
    
    residual.add(-alpha,w);
    
    rho[k] = sqr(residual.norm());
    
    display->Message(0,"CG residual rho = %.5f, rho/||b|| = %.5f, tol = %.5f",rho[k-1],rho[k-1]/norm_b,tol);
    
    if ( sqrt(rho[k-1]) < tol*norm_b ){
      display->Message(0,"GMRES iteration converged, ||res|| = %f",rho[k-1]);
      break;
    }
    
  }

  norm_residual = GetResidual(x,b);

  display->Message(0,"Residual/||b|| = %f",norm_residual/norm_b);
  display->Message(0,"Residual = %f",norm_residual);

  delete rho;
}
//-----------------------------------------------------------------------------
void KrylovSolver::ApplyMatrix( Vector *x, Vector *Ax )
{
  A->Mult(x,Ax);
}
//-----------------------------------------------------------------------------
void KrylovSolver::ApplyMatrix( real **x, int comp )
{
  // Preconditioner
  SISolver sisolver;

  switch( pc ){ 
  case pc_richardson:
    sisolver.SetMethod(richardson);
    break;
  case pc_jacobi:
    sisolver.SetMethod(jacobi);
    break;
  case pc_gaussseidel:
    sisolver.SetMethod(gaussseidel);
    break;
  case pc_sor:
    sisolver.SetMethod(sor);
    break;
  case pc_none:
    break;
  default:
    display->InternalError("SISolver::Solve()","Preconditioner not implemented");
  }

  Vector tmp1(A->Size(0)); 
  Vector tmp2(A->Size(0)); 

  for (int i=0; i<A->Size(0); i++)
	 tmp1(i) = x[i][comp];

  // Precondition 
  no_pc_sweeps = 1;
  if ( pc != pc_none ){
    sisolver.SetNoIterations(no_pc_sweeps);
    sisolver.Solve(A,&tmp2,&tmp1);
    tmp1 = tmp2;
  }      
    
  A->Mult(&tmp1,&tmp2);

  for (int i=0; i<A->Size(0); i++) x[i][comp+1] = tmp2(i);
}
//-----------------------------------------------------------------------------
void KrylovSolver::SetMatrix(SparseMatrix* A)
{
  this->A = A;
}
//-----------------------------------------------------------------------------
void KrylovSolver::SolvePxv( Vector *x )
{
  // Preconditioner
  SISolver sisolver;
  
  switch( pc ){ 
  case pc_richardson:
    sisolver.SetMethod(richardson);
    break;
  case pc_jacobi:
    sisolver.SetMethod(jacobi);
    break;
  case pc_gaussseidel:
    sisolver.SetMethod(gaussseidel);
    break;
  case pc_sor:
    sisolver.SetMethod(sor);
    break;
  case pc_none:
    break;
  default:
    display->InternalError("SISolver::Solve()","Preconditioner not implemented");
  }

  Vector tmp1(A->Size(0)); 

  for (int i=0; i<A->Size(0); i++)
	 tmp1(i) = (*x)(i);

  // Solve preconditioned problem 
  sisolver.SetNoIterations(no_pc_sweeps);
  sisolver.Solve(A,x,&tmp1);
}
//-----------------------------------------------------------------------------
void KrylovSolver::ComputeResidual( Vector *x, Vector *b, Vector *res )
{
  real Axrow;
  for (int i=0;i<A->Size(0);i++){
    Axrow  = A->Mult(i,x);
    (*res)(i) = (*b)(i) - Axrow;
  }
}
//-----------------------------------------------------------------------------
real KrylovSolver::GetResidual( Vector *x, Vector *b )
{
  real Axrow;
  norm_residual = 0.0;
  for (int i=0;i<A->Size(0);i++){
    Axrow  = A->Mult(i,x);
    norm_residual += sqr( (*b)(i) - Axrow );
  }
  return sqrt(norm_residual);
}
//-----------------------------------------------------------------------------
bool KrylovSolver::TestForOrthogonality( real **v )
{
  // if (||Av_k||+delta*||v_k+1||=||Av_k||)  delta \approx 10^-3

  display->Message(5,"Reorthogonalization needed");

  return false;
}
//-----------------------------------------------------------------------------
void KrylovSolver::AllocateArrays(int n, int k_max)
{
  mat_H = new(real *)[k_max+1];
  for (int i=0;i<k_max+1; i++)
    mat_H[i] = new real[k_max+1];
  for (int i=0;i<k_max+1; i++){
    for (int j = 0; j < k_max+1; j++) mat_H[i][j] = 0.0;
  }

  mat_r = new(real *)[k_max+1];
  for (int i = 0; i < k_max+1; i++)
    mat_r[i] = new real[k_max+1];
  for (int i = 0; i < k_max+1; i++){
    for (int j = 0; j < k_max+1; j++) mat_r[i][j] = 0.0;
  }

  mat_v = new(real *)[n];
  for (int i=0;i<n; i++)
    mat_v[i] = new real[k_max+1];
  for (int i=0;i<n; i++){
    for (int j = 0; j < k_max+1; j++) mat_v[i][j] = 0.0;
  }
  
  vec_s = new real[k_max+1]; 
  for (int i=0;i<k_max+1;i++) vec_s[i] = 0.0;

  vec_y = new real[k_max+1]; 
  for (int i=0;i<k_max+1;i++) vec_y[i] = 0.0;

  vec_w = new real[k_max+1]; 
  for (int i=0;i<k_max+1;i++) vec_w[i] = 0.0;

  vec_c = new real[k_max+1];
  for (int i=0;i<k_max+1;i++) vec_c[i] = 0.0;

  vec_g = new real[k_max+1]; 
  for (int i=0;i<k_max+1;i++) vec_g[i] = 0.0;

}
//-----------------------------------------------------------------------------
void KrylovSolver::DeleteArrays(int n, int k_max)
{
  for (int i=0;i<k_max+1;i++)
    delete [] mat_H[i];
  delete [] mat_H;
  
  for (int i=0;i<k_max+1;i++)
    delete [] mat_r[i];
  delete [] mat_r;
  
  for (int i=0;i<n;i++)
    delete [] mat_v[i];
  delete [] mat_v;

  delete [] vec_g;
  delete [] vec_c;
  delete [] vec_s;
  delete [] vec_y;
  delete [] vec_w;
}
//-----------------------------------------------------------------------------
