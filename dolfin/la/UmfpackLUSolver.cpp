// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Dag Lindbo 2008.
// 
// First added:  2006-06-01
// Last changed: 2008-07-19

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/timing.h>
#include "UmfpackLUSolver.h"
#include "uBlasKrylovSolver.h"
#include "uBlasSparseMatrix.h"
#include "uBlasKrylovMatrix.h"
#include "uBlasVector.h"

extern "C" 
{
#ifdef HAS_UMFPACK
#include <umfpack.h>
#endif
}

using namespace dolfin;

//-----------------------------------------------------------------------------
UmfpackLUSolver::UmfpackLUSolver() : AA(0), ej(0), Aj(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::~UmfpackLUSolver()
{
  if ( AA ) 
    delete AA;
  if ( ej ) 
    delete ej;
  if ( Aj ) 
    delete Aj;
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::solve(const uBlasMatrix<ublas_dense_matrix>& A, 
                                  uBlasVector& x, const uBlasVector& b)
{    
  // Get underlying uBLAS vectors
  ublas_vector& _x = x.vec(); 
  const ublas_vector& _b = b.vec(); 

  // Make copy of matrix and vector
  ublas_dense_matrix Atemp(A.mat());
  _x.resize(_b.size());
  _x.assign(_b);

  // Solve
  return solveInPlace(Atemp, _x);
}
//-----------------------------------------------------------------------------
#ifdef HAS_UMFPACK
dolfin::uint UmfpackLUSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, 
                                    uBlasVector& x, const uBlasVector& b)
{
  // Factorize matrix
  factorize(A);

  // Solve system
  factorizedSolve(x, b);

  // Clear data
  umfpack.clear();

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::factorize(const uBlasMatrix<ublas_sparse_matrix>& A)
{
  // Check dimensions and get number of non-zeroes
  const uint M  = A.size(0);
  const uint N  = A.size(1);
  const uint nz = A.mat().nnz();
  dolfin_assert(M == N);
  dolfin_assert(nz >= N); 

  // Make sure matrix assembly is complete
  (const_cast< ublas_sparse_matrix& >(A.mat())).complete_index1_data(); 

  // Allocate umfpack data
  umfpack.N  = M;
  umfpack.Rp = new long int[M+1];
  umfpack.Ri = new long int[nz];
  umfpack.Rx = new double[nz];

  if(umfpack.factorized)
  {
    warning("LUSolver already contains a factorized matrix! Clearing and starting over.");
    umfpack.clear();
  }
    
  // Transpose of A and set umfpack data
  umfpack.transpose((const std::size_t*) &A.mat().index1_data()[0], 
                    (const std::size_t*) &A.mat().index2_data()[0], &A.mat().value_data()[0]);

  // Factorize
  message("LU-factorizing linear system of size %d x %d (UMFPACK).", M, M);
  umfpack.factorize();
  umfpack.factorized = true;
  umfpack.mat_dim = M;

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::factorizedSolve(uBlasVector& x, const uBlasVector& b)
{
  const uint N  = b.size();

  if(!umfpack.factorized)
    error("Factorized solve must be preceeded by call to factorize.");

  if(N != umfpack.mat_dim)
    error("Vector does not match size of factored matrix");

  // Initialise solution vector and solve
  x.init(N);
  message("Solving factorized linear system of size %d x %d (UMFPACK).", N, N);
  umfpack.factorizedSolve(&x.vec().data()[0], &b.vec().data()[0]);

  return 1;
}
//-----------------------------------------------------------------------------
#else
dolfin::uint UmfpackLUSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, 
                                    uBlasVector& x, const uBlasVector& b)
{
  warning("UMFPACK must be installed to peform a LU solve for uBlas matrices. A Krylov iterative solver will be used instead.");

  uBlasKrylovSolver solver;
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::factorize(const uBlasMatrix<ublas_sparse_matrix>& A)
{
  error("UMFPACK must be installed to perform sparse LU factorization.");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::factorizedSolve(uBlasVector& x, const uBlasVector& b)
{
  error("UMFPACK must be installed to perform sparse back and forward substitution");
  return 0;
}
#endif
//-----------------------------------------------------------------------------
void UmfpackLUSolver::solve(const uBlasKrylovMatrix& A, uBlasVector& x,
			                      const uBlasVector& b)
{
  // The linear system is solved by computing a dense copy of the matrix,
  // obtained through multiplication with unit vectors.

  // Check dimensions
  const uint M  = A.size(0);
  const uint N  = A.size(1);
  dolfin_assert(M == N);
  dolfin_assert(M == b.size());

  // Initialize temporary data if not already done
  if ( !AA )
  {
    AA = new uBlasMatrix<ublas_dense_matrix>(M, N);
    ej = new uBlasVector(N);
    Aj = new uBlasVector(M);
  }
  else
  {
    AA->init(M, N);
    ej->init(N);
    Aj->init(N);
  }

  // Get underlying uBLAS vectors
  ublas_vector& _ej = ej->vec(); 
  ublas_vector& _Aj = Aj->vec(); 
  ublas_dense_matrix& _AA = AA->mat(); 

  // Reset unit vector
  _ej *= 0.0;

  // Compute columns of matrix
  for (uint j = 0; j < N; j++)
  {
    (_ej)(j) = 1.0;

    // Compute product Aj = Aej
    A.mult(*ej, *Aj);
    
    // Set column of A
    column(_AA, j) = _Aj;
    
    (_ej)(j) = 0.0;
  }

  // Solve linear system
  solve(*AA, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::solveInPlaceUBlas(uBlasMatrix<ublas_dense_matrix>& A, 
					      uBlasVector& x, const uBlasVector& b) const
{
  const uint M = A.size(0);
  dolfin_assert(M == b.size());
  
  // Get underlying uBLAS vectors
  ublas_vector& _x = x.vec(); 
  const ublas_vector& _b = b.vec(); 

  if( _x.size() != M )
    _x.resize(M);

  // Initialise solution vector
  _x.assign(_b);

  // Solve
  return solveInPlace(A.mat(), _x);
}
//-----------------------------------------------------------------------------
// UmfpackLUSolver::Umfpack implementation
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::clear()
{
  delete dnull; dnull = 0;
  delete inull; inull = 0;
  if(Symbolic)
  {
#ifdef HAS_UMFPACK
    umfpack_dl_free_symbolic(&Symbolic);
#endif
    Symbolic = 0;
  }
  if(Numeric)
  {
#ifdef HAS_UMFPACK
    umfpack_dl_free_numeric(&Numeric);
#endif
    Numeric = 0;
  }
  delete [] Rp; Rp = 0;
  delete [] Ri; Ri = 0;
  delete [] Rx; Rx = 0;
  factorized =  false;
  mat_dim = 0;
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::transpose(const std::size_t* Ap, 
                                      const std::size_t* Ai, const double* Ax)
{  
#ifdef HAS_UMFPACK
  long int status = umfpack_dl_transpose(N, N, (const long int*) Ap, 
                           (const long int*) Ai, Ax, inull, inull, Rp, Ri, Rx);
  Umfpack::checkStatus(status, "transpose");
#else
  error("UMFPACK not installed");
#endif
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::factorize()
{
  dolfin_assert(Rp);
  dolfin_assert(Ri);
  dolfin_assert(Rx);

#ifdef HAS_UMFPACK
  long int status;

  // Symbolic step (reordering etc)
  status= umfpack_dl_symbolic(N, N, (const long int*) Rp,(const long int*) Ri, 
                              Rx, &Symbolic, dnull, dnull);
  checkStatus(status, "symbolic");

  // Factorization step
  status = umfpack_dl_numeric((const long int*) Rp,(const long int*) Ri, Rx, 
                               Symbolic, &Numeric, dnull, dnull);
  Umfpack::checkStatus(status, "numeric");

  // Discard the symbolic part (since the factorization is complete.)
  umfpack_dl_free_symbolic(&Symbolic);
  Symbolic = 0;
#else
  error("UMFPACK not installed");
#endif
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::factorizedSolve(double*x, const double* b)
{
  dolfin_assert(Rp);
  dolfin_assert(Ri);
  dolfin_assert(Rx);
  dolfin_assert(Numeric);

#ifdef HAS_UMFPACK
  long int status  = umfpack_dl_solve(UMFPACK_A, (const long int*) Rp, 
                                     (const long int*) Ri, Rx, x, b, Numeric, 
                                     dnull, dnull);
  Umfpack::checkStatus(status, "solve");
#else
  error("UMFPACK not installed");
#endif
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::checkStatus(long int status, std::string function)
{
#ifdef HAS_UMFPACK
  if(status == UMFPACK_OK)
    return;

  // Help out by printing which UMFPACK function is returning the warning/error
  cout << "UMFPACK problem related to call to " << function << endl;

  if(status == UMFPACK_WARNING_singular_matrix)
    warning("UMFPACK reports that the matrix being solved is singular.");
  else if(status == UMFPACK_ERROR_out_of_memory)
    error("UMFPACK has run out of memory solving a system.");
  else if(status == UMFPACK_ERROR_invalid_system)
    error("UMFPACK reports an invalid system. Is the matrix square?.");
  else if(status == UMFPACK_ERROR_invalid_Numeric_object)
    error("UMFPACK reports an invalid Numeric object.");
  else if(status == UMFPACK_ERROR_invalid_Symbolic_object)
    error("UMFPACK reports an invalid Symbolic object.");
  else if(status != UMFPACK_OK)
    warning("UMFPACK is reporting an unknown error.");

#else
  error("Problem with DOLFIN build configuration for using UMFPACK.");   
#endif
}
//-----------------------------------------------------------------------------



