// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Dag Lindbo 2008.
// 
// First added:  2006-06-01
// Last changed: 2008-07-21

#include <dolfin/log/dolfin_log.h>
#include "UmfpackLUSolver.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "KrylovSolver.h"
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
#ifdef HAS_UMFPACK
dolfin::uint UmfpackLUSolver::solve(const GenericMatrix& A, GenericVector& x, 
                                    const GenericVector& b)
{
  // Factorize matrix
  factorize(A);

  // Solve system
  factorizedSolve(x, b);

  // Clear data
  umfpack.clear();

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::factorize(const GenericMatrix& A)
{
  // Check dimensions and get number of non-zeroes
  boost::tuple<const std::size_t*, const std::size_t*, const double*, int> data = A.data();
  const uint M   = A.size(0);
  const uint nnz = data.get<3>();
  dolfin_assert(A.size(0) == A.size(1));

  dolfin_assert(nnz >= M); 

  // Initialise umfpack data
  umfpack.init((const long int*) data.get<0>(), (const long int*) data.get<1>(), 
                data.get<2>(), M, nnz);

  // Factorize
  message("LU-factorizing linear system of size %d x %d (UMFPACK).", M, M);
  umfpack.factorize();

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::factorizedSolve(GenericVector& x, const GenericVector& b) const
{
  const uint N = b.size();

  if(!umfpack.factorized)
    error("Factorized solve must be preceeded by call to factorize.");

  if(N != umfpack.N)
    error("Vector does not match size of factored matrix");

  // Initialise solution vector and solve
  x.init(N);

  message("Solving factorized linear system of size %d x %d (UMFPACK).", N, N);
  // Solve for tranpose since we use compressed rows and UMFPACK expected compressed columns
  umfpack.factorizedSolve(x.data(), b.data(), true);

  return 1;
}
//-----------------------------------------------------------------------------
#else
dolfin::uint UmfpackLUSolver::solve(const GenericMatrix& A, GenericVector& x, 
                                    const GenericVector& b)
{
  warning("UMFPACK must be installed to peform a LU solve for uBlas matrices. A Krylov iterative solver will be used instead.");

  KrylovSolver solver;
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::factorize(const GenericMatrix& A)
{
  error("UMFPACK must be installed to perform sparse LU factorization.");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::factorizedSolve(GenericVector& x, 
                                              const GenericVector& b) const
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
  warning("UmfpackLUSolver no longer solves dense matrices. This function will be removed and probably added to uBlasKrylovSolver.");
  warning("The uBlasKrylovSolver LU solver has been modified and has not yet been well tested. Please verify your results.");
 (*AA).solve(x, b);
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
  if(local_matrix)
  {
    delete [] Rp; Rp = 0;
    delete [] Ri; Ri = 0;
    delete [] Rx; Rx = 0;
    local_matrix = false;
  }
  factorized =  false;
  N = 0;
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::init(const long int* Ap, const long int* Ai, 
                                         const double* Ax, uint M, uint nz)
{  
  if(factorized)
    warning("LUSolver already contains a factorized matrix! Clearing and starting over.");

  // Clear any data
  clear();

  // Set umfpack data
  N  = M;
  Rp = Ap;
  Ri = Ai;
  Rx = Ax;
  local_matrix = false;
  N = M;
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::initTranspose(const long int* Ap, const long int* Ai, 
                                         const double* Ax, uint M, uint nz)
{  
#ifdef HAS_UMFPACK
  if(Rp || Ri || Rx)
    error("UmfpackLUSolver data already points to a matrix");

  // Allocate memory and take ownership
  clear();
  Rp = new long int[M+1];
  Ri = new long int[nz];
  Rx = new double[nz];
  local_matrix = true;
  N  = M;

  // Compute transpse
  long int status = umfpack_dl_transpose(M, M, Ap, Ai, Ax, inull, inull, 
                    const_cast<long int*>(Rp), const_cast<long int*>(Ri), const_cast<double*>(Rx));
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
  dolfin_assert(!Symbolic);
  dolfin_assert(!Numeric);

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

  factorized = true;
#else
  error("UMFPACK not installed");
#endif
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::factorizedSolve(double*x, const double* b, bool transpose) const
{
  dolfin_assert(Rp);
  dolfin_assert(Ri);
  dolfin_assert(Rx);
  dolfin_assert(Numeric);

#ifdef HAS_UMFPACK
  long int status;
  if(transpose)
    status = umfpack_dl_solve(UMFPACK_At, Rp, Ri, Rx, x, b, Numeric, dnull, dnull);
  else
    status = umfpack_dl_solve(UMFPACK_A, Rp, Ri, Rx, x, b, Numeric, dnull, dnull);

  Umfpack::checkStatus(status, "solve");
#else
  error("UMFPACK not installed");
#endif
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::checkStatus(long int status, std::string function) const
{
#ifdef HAS_UMFPACK
  if(status == UMFPACK_OK)
    return;

  // Printing which UMFPACK function is returning an warning/error
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

