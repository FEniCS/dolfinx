// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Dag Lindbo 2008.
//
// First added:  2006-06-01
// Last changed: 2008-07-08

#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
#include "UmfpackLUSolver.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "LUSolver.h"

extern "C"
{
#ifdef HAS_UMFPACK
#include <umfpack.h>
#endif
}

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters UmfpackLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("umfpack_lu_solver");
  return p;
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::UmfpackLUSolver() : symbolic(0), numeric(0)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::UmfpackLUSolver(const GenericMatrix& A)
                               : symbolic(0), numeric(0),
                                 A(reference_to_no_delete_pointer(A))
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::UmfpackLUSolver(boost::shared_ptr<const GenericMatrix> A)
                               : symbolic(0), numeric(0), A(A)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::~UmfpackLUSolver()
{
  clear();
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::set_operator(const GenericMatrix& A)
{
  clear();
  this->A = reference_to_no_delete_pointer(A);
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  // Factorize matrix
  factorize();

  // Solve
  return solve_factorized(x, b);
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::symbolic_factorize()
{
  if (!A)
    error("A matrix must be assocoated with UmfpackLUSolver to peform a symbolic factorisation.");

  // Get matrix data
  std::tr1::tuple<const std::size_t*, const std::size_t*, const double*, int> data = A->data();
  const std::size_t* Ap  = std::tr1::get<0>(data);
  const std::size_t* Ai  = std::tr1::get<1>(data);
  const double*      Ax  = std::tr1::get<2>(data);
  const uint         nnz = std::tr1::get<3>(data);

  // Check dimensions and get number of non-zeroes
  const uint M   = A->size(0);
  const uint N   = A->size(1);
  assert(nnz >= M);

  // Clear any old factorizations
  clear();

  // Factorize and solve
  info(PROGRESS, "Symbolic factorization of a matrix of size %d x %d (UMFPACK).", M, N);

  // Perform symbolic factorisation
  symbolic = umfpack_factorize_symbolic(M, N, (const long int*)Ap, (const long int*)Ai, Ax);
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::factorize()
{
  if (!A)
    error("A matrix must be assocoated with UmfpackLUSolver to peform a factorisation.");

  // Get matrix data
  std::tr1::tuple<const std::size_t*, const std::size_t*, const double*, int> data = A->data();
  const std::size_t* Ap  = std::tr1::get<0>(data);
  const std::size_t* Ai  = std::tr1::get<1>(data);
  const double*      Ax  = std::tr1::get<2>(data);
  const uint         nnz = std::tr1::get<3>(data);

  // Check dimensions and get number of non-zeroes
  const uint M   = A->size(0);
  const uint N   = A->size(1);
  assert(nnz >= M);

  // Factorize and solve
  info(PROGRESS, "LU factorization of a matrix of size %d x %d (UMFPACK).", M, N);

  // Create symbolic factoriation if required
  if (!symbolic || !parameters["same_nonzero_pattern"])
    symbolic_factorize();

  // Perform LU factorisation
  numeric = umfpack_factorize_numeric((const long int*)Ap, (const long int*)Ai, Ax, symbolic);
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::solve_factorized(GenericVector& x,
                                               const GenericVector& b) const
{
  if (!A)
    error("No matrix associated with UmfpackLUSolver. Cannot perform factorized_solve.");

  assert(A->size(0) == A->size(0));
  assert(A->size(0) == b.size());

  // Resize x if required
  if (A->size(1) != x.size())
    x.resize(A->size(1));

  if (!symbolic)
    error("No symbolic factorisation. Please call UmfpackLUSolver::factorize_symbolic().");

  if (!numeric)
    error("No LU factorisation. Please call UmfpackLUSolver::factorize_numeric().");

  // Get matrix data
  std::tr1::tuple<const std::size_t*, const std::size_t*, const double*, int> data = A->data();
  const std::size_t* Ap  = std::tr1::get<0>(data);
  const std::size_t* Ai  = std::tr1::get<1>(data);
  const double*      Ax  = std::tr1::get<2>(data);

  // Solve for tranpose since we use compressed rows and UMFPACK expected compressed columns
  umfpack_solve((const long int*)Ap, (const long int*)Ai, Ax, x.data(), b.data(), numeric);

  return 1;
}
//-----------------------------------------------------------------------------
#ifdef HAS_UMFPACK
//-----------------------------------------------------------------------------
void UmfpackLUSolver::clear()
{
  if (numeric)
  {
    umfpack_dl_free_numeric(&numeric);
    numeric = 0;
  }

  if(symbolic)
  {
    umfpack_dl_free_symbolic(&symbolic);
    symbolic = 0;
  }
}
//-----------------------------------------------------------------------------
void* UmfpackLUSolver::umfpack_factorize_symbolic(uint M, uint N,
                                                   const long int* Ap,
                                                   const long int* Ai,
                                                   const double* Ax)
{
  assert(Ap);
  assert(Ai);
  assert(Ax);

  void* symbolic = 0;
  double* dnull = 0;

  // Symbolic factorisation step (reordering, etc)
  long int status = umfpack_dl_symbolic(M, N, Ap, Ai, Ax, &symbolic, dnull, dnull);
  delete dnull;

  umfpack_check_status(status, "symbolic");

  return symbolic;
}
//-----------------------------------------------------------------------------
void* UmfpackLUSolver::umfpack_factorize_numeric(const long int* Ap,
                                                  const long int* Ai,
                                                  const double* Ax,
                                                  void* symbolic)
{
  assert(Ap);
  assert(Ai);
  assert(Ax);
  assert(symbolic);

  void* numeric = 0;
  double* dnull = 0;

  // Factorization step
  long int status = umfpack_dl_numeric(Ap, Ai, Ax, symbolic, &numeric, dnull, dnull);
  delete dnull;

  umfpack_check_status(status, "numeric");

  return numeric;
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::umfpack_solve(const long int* Ap, const long int* Ai,
                                    const double* Ax, double* x,
                                    const double* b, void* numeric)
{
  assert(Ap);
  assert(Ai);
  assert(Ax);
  assert(x);
  assert(b);
  assert(numeric);
  double* dnull = 0;

  // Solve system. We assume CSR storage, but UMFPACK expects CSC, so solve
  // for the transpose
  long int status = umfpack_dl_solve(UMFPACK_At, Ap, Ai, Ax, x, b, numeric, dnull, dnull);
  delete dnull;

  umfpack_check_status(status, "solve");
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::umfpack_check_status(long int status,
                                            std::string function)
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
//-----------------------------------------------------------------------------
#else
//-----------------------------------------------------------------------------
void UmfpackLUSolver::clear()
{
  error("Umfpack not installed. Cannot perform LU solver using Umfpack.");
}
//-----------------------------------------------------------------------------
void* UmfpackLUSolver::umfpack_factorize_symbolic(uint M, uint N,
                                                   const long int* Ap,
                                                   const long int* Ai,
                                                   const double* Ax)
{
  error("Umfpack not installed. Cannot perform LU solver using Umfpack.");
  return 0;
}
//-----------------------------------------------------------------------------
void* UmfpackLUSolver::umfpack_factorize_numeric(const long int* Ap,
                                                  const long int* Ai,
                                                  const double* Ax,
                                                  void* symbolic)
{
  error("Umfpack not installed. Cannot perform LU solver using Umfpack.");
  return 0;
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::umfpack_solve(const long int* Ap, const long int* Ai,
                                    const double* Ax, double* x,
                                    const double* b, void* numeric)
{
  error("Umfpack not installed. Cannot perform LU solver using Umfpack.");
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::umfpack_check_status(long int status,
                                            std::string function)
{
  error("Umfpack not installed. Cannot perform LU solver using Umfpack.");
}
//-----------------------------------------------------------------------------
#endif
//-----------------------------------------------------------------------------

