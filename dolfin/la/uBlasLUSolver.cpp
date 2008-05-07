// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Dah Lindbo 2008.
// 
// First added:  2006-06-01
// Last changed: 2008-05-07

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/timing.h>
#include "uBlasLUSolver.h"
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
uBlasLUSolver::uBlasLUSolver() : AA(0), ej(0), Aj(0)
{
  // Do nothing
#ifdef HAS_UMFPACK
  Rp = 0;
  Ri = 0;
  Rx = 0;
  has_factorized_matrix = false;
#endif
}
//-----------------------------------------------------------------------------
uBlasLUSolver::~uBlasLUSolver()
{
  if ( AA ) 
    delete AA;
  if ( ej ) 
    delete ej;
  if ( Aj ) 
    delete Aj;

#ifdef HAS_UMFPACK
   if(Rp) 
    delete [] Rp;
   if(Ri) 
    delete [] Ri;
   if(Rx)
   { 
      delete [] Rx;
      umfpack_dl_free_numeric(&Numeric);
   }
#endif
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_dense_matrix>& A, 
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
dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, 
                                  const uBlasVector& b)
{
  factorize(A);
  factorized_solve(x,b);

  umfpack_dl_free_numeric(&Numeric);
  delete [] Rp; Rp = 0;
  delete [] Ri; Ri = 0;
  delete [] Rx; Rx = 0;
  has_factorized_matrix = false;

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::factorize(const uBlasMatrix<ublas_sparse_matrix>& A)
{
// Symbolic and numeric part of UMFPACK solve procedure

  const ublas_sparse_matrix& _A = A.mat(); 

  // Check dimensions and get number of non-zeroes
  const uint M  = A.size(0);
  const uint N  = A.size(1);
  const uint nz = _A.nnz();

  dolfin_assert(M == N);
  dolfin_assert(nz >= N); 

  // Make sure matrix assembly is complete
  (const_cast< ublas_sparse_matrix& >(_A)).complete_index1_data(); 

  message("LU-factorizing linear system of size %d x %d (UMFPACK).", M, M);

  double* dnull = (double *) NULL;
  long int* inull = (long int *) NULL;
  void *Symbolic;
  const std::size_t* Ap = &(_A.index1_data() [0]);
  const std::size_t* Ai = &(_A.index2_data() [0]);
  const double* Ax = &(_A.value_data() [0]);

  if(has_factorized_matrix)
  {
    warning("LUSolver already contains a factorized matrix! Clearing and starting over.");
    umfpack_dl_free_numeric(&Numeric);
    delete [] Rp;
    delete [] Ri;
    delete [] Rx;
  }
  
  Rp = new long int[M+1];
  Ri = new long int[nz];
  Rx = new double[nz];
  
  long int status;
  
  // Transpose of A
  status= umfpack_dl_transpose(M, M, (const long int*) Ap, (const long int*) Ai, Ax, inull, inull, Rp, Ri, Rx);
  check_status(status, "transpose");

  // Symbolic step (reordering etc)
  status= umfpack_dl_symbolic(M, M, (const long int*) Rp, (const long int*) Ri, Rx, &Symbolic, dnull, dnull);
  check_status(status, "symbolic");

  // Factorization step
  status = umfpack_dl_numeric( (const long int*) Rp, (const long int*) Ri, Rx, Symbolic, &Numeric, dnull, dnull);
  check_status(status, "numeric");

  // Discard the symbolic part (since the factorization is complete.)
  umfpack_dl_free_symbolic(&Symbolic);

  has_factorized_matrix = true;
  mat_dim = M;

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::factorized_solve(uBlasVector& x, const uBlasVector& b)
{
  const uint N  = b.size();

  if(!has_factorized_matrix)
    error("Factorized solve must be preceeded by call to factorize.");

  if(N != mat_dim)
    error("Vector does not match size of factored matrix");

  x.init(N);
  
  message("Solving factorized linear system of size %d x %d (UMFPACK).", N, N);

  ublas_vector& _x = x.vec(); 
  const ublas_vector& _b = b.vec(); 
  double* dnull = (double *) NULL;
  double* xx = &(_x.data() [0]);
  const double* bb = &(_b.data() [0]);

  long int status;

  status = umfpack_dl_solve(UMFPACK_A, (const long int*) Rp, (const long int*) Ri, Rx, xx, bb, Numeric, dnull, dnull);
  check_status(status, "solve");
  
  return 1;
}
//-----------------------------------------------------------------------------
#else
dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, 
				  const uBlasVector& b)
{
  warning("UMFPACK must be installed to peform a LU solve for uBlas matrices. A Krylov iterative solver will be used instead.");

  uBlasKrylovSolver solver;
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::factorize(const uBlasMatrix<ublas_sparse_matrix>& A)
{
  error("UMFPACK must be installed to perform sparse LU factorization.");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::factorized_solve(uBlasVector& x, const uBlasVector& b)
{
  error("UMFPACK must be installed to perform sparse back and forward substitution");
  return 0;
}
#endif
//-----------------------------------------------------------------------------
void uBlasLUSolver::solve(const uBlasKrylovMatrix& A, uBlasVector& x,
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
dolfin::uint uBlasLUSolver::solveInPlaceUBlas(uBlasMatrix<ublas_dense_matrix>& A, 
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
void uBlasLUSolver::check_status(long int status, std::string function) const
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



