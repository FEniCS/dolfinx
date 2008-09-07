// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Dag Lindbo 2008.
// 
// First added:  2006-06-01
// Last changed: 2008-09-05

#include <dolfin/log/dolfin_log.h>
#include "UmfpackLUSolver.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "uBLASKrylovMatrix.h"
#include "uBLASVector.h"

extern "C" 
{
#ifdef HAS_UMFPACK
#include <umfpack.h>
#endif
}

using namespace dolfin;

//-----------------------------------------------------------------------------
UmfpackLUSolver::UmfpackLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::~UmfpackLUSolver()
{
  // Do nothing
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
  std::tr1::tuple<const std::size_t*, const std::size_t*, const double*, int> data = A.data();
  const uint M   = A.size(0);
  const uint nnz = std::tr1::get<3>(data);
  dolfin_assert(A.size(0) == A.size(1));

  dolfin_assert(nnz >= M); 

  // Initialise umfpack data
  umfpack.init((const long int*) std::tr1::get<0>(data), 
    (const long int*) std::tr1::get<1>(data), std::tr1::get<2>(data), M, nnz);

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
  warning("UMFPACK must be installed to peform a LU solve for uBLAS matrices. A Krylov iterative solver will be used instead.");

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
  error("UMFPACK must be installed to perform sparse backward and forward substitutions.");
  return 0;
}
#endif
//-----------------------------------------------------------------------------
#ifdef HAS_UMFPACK
// UmfpackLUSolver::Umfpack implementation
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::clear()
{
  delete dnull; dnull = 0;
  delete inull; inull = 0;
  if(Symbolic)
  {
    umfpack_dl_free_symbolic(&Symbolic);
    Symbolic = 0;
  }
  if(Numeric)
  {
    umfpack_dl_free_numeric(&Numeric);
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
  N  = M;
  local_matrix = false;
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::initTranspose(const long int* Ap, const long int* Ai, 
                                         const double* Ax, uint M, uint nz)
{  
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
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::factorize()
{
  dolfin_assert(Rp);
  dolfin_assert(Ri);
  dolfin_assert(Rx);
  dolfin_assert(!Symbolic);
  dolfin_assert(!Numeric);

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
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::factorizedSolve(double*x, const double* b, bool transpose) const
{
  dolfin_assert(Rp);
  dolfin_assert(Ri);
  dolfin_assert(Rx);
  dolfin_assert(Numeric);

  long int status;
  if(transpose)
    status = umfpack_dl_solve(UMFPACK_At, Rp, Ri, Rx, x, b, Numeric, dnull, dnull);
  else
    status = umfpack_dl_solve(UMFPACK_A, Rp, Ri, Rx, x, b, Numeric, dnull, dnull);

  Umfpack::checkStatus(status, "solve");
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::Umfpack::checkStatus(long int status, std::string function) const
{
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
}
#endif
//-----------------------------------------------------------------------------

