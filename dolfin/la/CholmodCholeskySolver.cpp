// Copyright (C) 2008 Dag Lindbo and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-15
// Last changed: 2008-08-17

#include <cstring>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/KrylovSolver.h>
#include "UmfpackLUSolver.h"
#include "CholmodCholeskySolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CholmodCholeskySolver::CholmodCholeskySolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CholmodCholeskySolver::~CholmodCholeskySolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
//=============================================================================
#ifdef HAS_CHOLMOD
dolfin::uint CholmodCholeskySolver::solve(const GenericMatrix& A,
					  GenericVector& x, 
					  const GenericVector& b)
{
  // Factorize matrix
  factorize(A);

  // Solve system
  factorizedSolve(x, b);

  // Clear data
  cholmod.clear();

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint CholmodCholeskySolver::factorize(const GenericMatrix& A)
{
  // Check dimensions and get number of non-zeroes
  std::tr1::tuple<const std::size_t*, const std::size_t*, const double*, int> data = A.data();
  const uint M   = A.size(0);
  const uint nnz = std::tr1::get<3>(data);
  dolfin_assert(A.size(0) == A.size(1));

  dolfin_assert(nnz >= M); 

  // Initialise cholmod data
  // NOTE: Casting away const here
  cholmod.init((UF_long*) std::tr1::get<0>(data),(UF_long*) std::tr1::get<1>(data), 
	             (double*) std::tr1::get<2>(data), M, nnz);

  // Factorize
  message("Cholesky-factorizing linear system of size %d x %d (CHOLMOD).",M,M);
  cholmod.factorize();

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint CholmodCholeskySolver::factorizedSolve(GenericVector& x, const GenericVector& b)
{
  const uint N = b.size();

  if(!cholmod.factorized)
    error("Factorized solve must be preceeded by call to factorize.");

  if(N != cholmod.N)
    error("Vector does not match size of factored matrix");

  // Initialise solution vector and solve
  x.resize(N);

  message("Solving factorized linear system of size %d x %d (CHOLMOD).", N, N);

  cholmod.factorizedSolve(x.data(), b.data());

  return 1;
}
//-----------------------------------------------------------------------------
#else
// ============================================================================
dolfin::uint CholmodCholeskySolver::solve(const GenericMatrix& A, 
					  GenericVector& x, 
					  const GenericVector& b)
{
  warning("CHOLMOD must be installed to peform a Cholesky solve for the current backend. Attemping to use UMFPACK solver.");

  UmfpackLUSolver solver;
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint CholmodCholeskySolver::factorize(const GenericMatrix& A)
{
  error("CHOLMOD must be installed to perform sparse Cholesky factorization.");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint CholmodCholeskySolver::factorizedSolve(GenericVector& x, const GenericVector& b)
{
  error("CHOLMOD must be installed to perform sparse back and forward substitution");
  return 0;
}
#endif

//==============================================================================
// CholmodCholeskySolver::Cholmod implementation
#ifdef HAS_CHOLMOD
//==============================================================================

CholmodCholeskySolver::Cholmod::Cholmod() : 
  N(0), factorized(false), A_chol(0), L_chol(0)
{
  // "Start" cholmod
  cholmod_l_start(&c);
}
//-----------------------------------------------------------------------------
CholmodCholeskySolver::Cholmod::~Cholmod() 
{ 
  clear(); 
	
  // "stop" cholmod
  cholmod_l_finish(&c);
}
//-----------------------------------------------------------------------------
void CholmodCholeskySolver::Cholmod::clear()
{
  if(A_chol)
  {
    cholmod_l_free(1, sizeof(cholmod_sparse), A_chol, &c);
    A_chol = 0;
  }
  if(L_chol)
  {
    cholmod_l_free_factor(&L_chol, &c);
    L_chol = 0;
  }
}
//-----------------------------------------------------------------------------
void CholmodCholeskySolver::Cholmod::init(UF_long* Ap, UF_long* Ai, double* Ax, 
					  uint M, uint nz)
{ 
  if(factorized)
    warning("CholeskySolver already contains a factorized matrix! Clearing and starting over.");

  // Clear any data
  clear();

  A_chol = (cholmod_sparse*) cholmod_l_malloc(1, sizeof(cholmod_sparse), &c);

  // The matrix
  A_chol->p = Ap;
  A_chol->i = Ai;
  A_chol->x = Ax;

  A_chol->nzmax = nz;
  A_chol->ncol = M;
  A_chol->nrow = M;

  A_chol->sorted = 1;
  A_chol->packed = 1;

  A_chol->xtype = CHOLMOD_REAL;
  A_chol->stype = -1;
  A_chol->dtype = CHOLMOD_DOUBLE;
  A_chol->itype = CHOLMOD_LONG;

  N = M;
}
//-----------------------------------------------------------------------------
void CholmodCholeskySolver::Cholmod::factorize()
{
  // Analyze
  L_chol = cholmod_l_analyze(A_chol, &c);

  // Factorize
  cholmod_l_factorize(A_chol, L_chol, &c);

  checkStatus("factorize()");

  factorized = true;
}
//-----------------------------------------------------------------------------
void CholmodCholeskySolver::Cholmod::factorizedSolve(double*x, const double* b)
{
  cholmod_dense *x_chol, *b_chol;

  // initialize rhs
  b_chol = (cholmod_dense*) cholmod_l_malloc(1, sizeof(cholmod_dense), &c);
  b_chol->x = (double*) b;
  b_chol->nrow = N;
  b_chol->ncol = 1;
  b_chol->d = N;
  b_chol->nzmax = N;
  b_chol->xtype = A_chol->xtype;
  b_chol->dtype = A_chol->dtype;
  
  // Solve
  x_chol = cholmod_l_solve(CHOLMOD_A, L_chol, b_chol, &c);

  // Compute residual and residual norm
  cholmod_dense* r_chol = residual(x_chol,b_chol);
  double residn = residual_norm(r_chol, x_chol, b_chol);
  
  // Iterative refinement
  if(residn > 1.0e-14)
  {
    refine_once(x_chol, r_chol);

    cholmod_l_free_dense(&r_chol, &c);

    r_chol = residual(x_chol, b_chol);
    residn = residual_norm(r_chol, x_chol,b_chol);
  }

  // Solution vector
  // FIXME: Cholmod allocates its own solution vector.
  memcpy(x, x_chol->x, N*sizeof(double));
  cholmod_l_free_dense(&x_chol, &c);

  // Clear rhs
  cholmod_l_free(1, sizeof(cholmod_dense), b_chol, &c);

  // Clear residual
  cholmod_l_free_dense(&r_chol, &c);

  checkStatus("factorizedSolve()");
}
//-----------------------------------------------------------------------------
cholmod_dense* CholmodCholeskySolver::Cholmod::residual(cholmod_dense* x,
							cholmod_dense* b)
{
  double  one[2] = { 1, 0}; 
  double mone[2] = {-1, 0};
  
  // Residual r = r-A*x
  cholmod_dense* r = cholmod_l_copy_dense(b, &c);
  cholmod_l_sdmult(A_chol, 0, mone, one, x, r, &c);
  
  return r;
}
//-----------------------------------------------------------------------------
double CholmodCholeskySolver::Cholmod::residual_norm(cholmod_dense* r,
						     cholmod_dense* x,
						     cholmod_dense* b)
{
  double r_norm = cholmod_l_norm_dense(r, 0, &c);
  double x_norm = cholmod_l_norm_dense(x, 0, &c);
  double b_norm = cholmod_l_norm_dense(b, 0, &c);
  double A_norm = cholmod_l_norm_sparse(A_chol, 0, &c);
  double Axb_norm = A_norm*x_norm+b_norm;

  return r_norm/Axb_norm;
}
//-----------------------------------------------------------------------------
void CholmodCholeskySolver::Cholmod::refine_once(cholmod_dense* x, 
						 cholmod_dense* r)
{
  cholmod_dense* r_iter;
  r_iter = cholmod_l_solve(CHOLMOD_A, L_chol, r, &c); 
  
  double* xx = (double*) x->x;
  double* rx = (double*) r_iter->x;

  for(uint i = 0; i < N; i++)
    xx[i] = xx[i] + rx[i];

  cholmod_l_free_dense(&r_iter, &c);
}
//-----------------------------------------------------------------------------
void CholmodCholeskySolver::Cholmod::checkStatus(std::string function)
{
  UF_long status = c.status;

  if( status < 0)
  {
    cout << "\nCHOLMOD Warning: problem related to call to " << function
	    << ".\nFull CHOLMOD common dump:" << endl;

    cholmod_l_print_common(NULL, &c);
  }
  else if(status > 0)
  {
    cout << "\nCHOLMOD Fatal error: problem related to call to " << function
	    << ".\nFull CHOLMOD common dump:" << endl;
    cholmod_l_print_common(NULL, &c);
  }
}
//-----------------------------------------------------------------------------
#endif

