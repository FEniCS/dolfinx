// Copyright (C) 2006-2008 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2006.
// Modified by Dag Lindbo 2008.
//
// First added:  2006-06-01
// Last changed: 2011-03-24

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

#ifdef HAS_UMFPACK
namespace dolfin
{
  class UmfpackIntSymbolicDeleter
  {
  public:
    void operator() (void* symbolic)
    {
      if (symbolic)
      {
        umfpack_di_free_symbolic(&symbolic);
        symbolic = 0;
      }
    }
  };

  class UmfpackLongIntSymbolicDeleter
  {
  public:
    void operator() (void* symbolic)
    {
      if (symbolic)
      {
        umfpack_dl_free_symbolic(&symbolic);
        symbolic = 0;
      }
    }
  };

  class UmfpackIntNumericDeleter
  {
  public:
    void operator() (void* numeric)
    {
      if (numeric)
      {
        umfpack_di_free_numeric(&numeric);
        numeric = 0;
      }
    }
  };

  class UmfpackLongIntNumericDeleter
  {
  public:
    void operator() (void* numeric)
    {
      if (numeric)
      {
        umfpack_dl_free_numeric(&numeric);
        numeric = 0;
      }
    }
  };
}
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters UmfpackLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("umfpack_lu_solver");
  return p;
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::UmfpackLUSolver()
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::UmfpackLUSolver(const GenericMatrix& A)
                               : A(reference_to_no_delete_pointer(A))
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::UmfpackLUSolver(boost::shared_ptr<const GenericMatrix> A)
                               : A(A)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
UmfpackLUSolver::~UmfpackLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::set_operator(const GenericMatrix& A)
{
  symbolic.reset();
  numeric.reset();
  this->A = reference_to_no_delete_pointer(A);
}
//-----------------------------------------------------------------------------
const GenericMatrix& UmfpackLUSolver::get_operator() const
{
  if (!A)
    error("Operator for linear solver has not been set.");
  return *A;
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(A);

  // Get some parameters
  const bool reuse_fact   = parameters["reuse_factorization"];
  const bool same_pattern = parameters["same_nonzero_pattern"];

  // Perform symbolic factorization if required
  if (!symbolic)
    symbolic_factorize();
  else if (!reuse_fact && !same_pattern)
    symbolic_factorize();

  // Perform numerical factorization if required
  if (!numeric)
    numeric_factorize();
  else if (!reuse_fact)
    numeric_factorize();

  // Solve
  return solve_factorized(x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint UmfpackLUSolver::solve(const GenericMatrix& A, GenericVector& x,
                                    const GenericVector& b)
{
  set_operator(A);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::symbolic_factorize()
{
  if (!A)
    error("A matrix must be assocoated with UmfpackLUSolver to peform a symbolic factorisation.");

  // Clear any old factorizations
  symbolic.reset();
  numeric.reset();

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
  log(PROGRESS, "Symbolic factorization of a matrix of size %d x %d (UMFPACK).", M, N);

  // Perform symbolic factorisation
  symbolic = umfpack_factorize_symbolic(M, N, Ap, Ai, Ax);
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::numeric_factorize()
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
  log(PROGRESS, "LU factorization of a matrix of size %d x %d (UMFPACK).", M, N);

  assert(symbolic);
  numeric.reset();

  // Perform LU factorisation
  numeric = umfpack_factorize_numeric(Ap, Ai, Ax, symbolic.get());
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

  log(PROGRESS, "Solving linear system of size %d x %d (UMFPACK LU solver).",
      A->size(0), A->size(1));

  // Solve for tranpose since we use compressed rows and UMFPACK expected compressed columns
  umfpack_solve(Ap, Ai, Ax, x.data(), b.data(), numeric.get());

  return 1;
}
//-----------------------------------------------------------------------------
#ifdef HAS_UMFPACK
//-----------------------------------------------------------------------------
boost::shared_ptr<void> UmfpackLUSolver::umfpack_factorize_symbolic(uint M, uint N,
                                                                    const std::size_t* Ap,
                                                                    const std::size_t* Ai,
                                                                    const double* Ax)
{
  assert(Ap);
  assert(Ai);
  assert(Ax);

  void* symbolic = 0;
  boost::scoped_ptr<double> dnull;

  // Symbolic factorisation step (reordering, etc)
  if (sizeof(std::size_t) == sizeof(int))
  {
    const int* _Ap = reinterpret_cast<const int*>(Ap);
    const int* _Ai = reinterpret_cast<const int*>(Ai);
    long int status = umfpack_di_symbolic(M, N, _Ap, _Ai, Ax, &symbolic, dnull.get(), dnull.get());
    umfpack_check_status(status, "symbolic");
    return boost::shared_ptr<void>(symbolic, UmfpackIntSymbolicDeleter());
  }
  else if (sizeof(std::size_t) == sizeof(UF_long))
  {
    const UF_long* _Ap = reinterpret_cast<const UF_long*>(Ap);
    const UF_long* _Ai = reinterpret_cast<const UF_long*>(Ai);
    long int status = umfpack_dl_symbolic(M, N, _Ap, _Ai, Ax, &symbolic, dnull.get(), dnull.get());
    umfpack_check_status(status, "symbolic");
    return boost::shared_ptr<void>(symbolic, UmfpackLongIntSymbolicDeleter());
  }
  else
    error("Could not determine correct types for casting integers to pass to UMFPACK.");

  return boost::shared_ptr<void>();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<void> UmfpackLUSolver::umfpack_factorize_numeric(const std::size_t* Ap,
                                                 const std::size_t* Ai,
                                                 const double* Ax,
                                                 void* symbolic)
{
  assert(Ap);
  assert(Ai);
  assert(Ax);
  assert(symbolic);

  void* numeric = 0;
  boost::scoped_ptr<double> dnull;

  // Factorization step
  long int status = 0;
  if (sizeof(std::size_t) == sizeof(int))
  {
    const int* _Ap = reinterpret_cast<const int*>(Ap);
    const int* _Ai = reinterpret_cast<const int*>(Ai);
    status = umfpack_di_numeric(_Ap, _Ai, Ax, symbolic, &numeric, dnull.get(), dnull.get());
    umfpack_check_status(status, "numeric");
    return boost::shared_ptr<void>(numeric, UmfpackIntNumericDeleter());
  }
  else if (sizeof(std::size_t) == sizeof(UF_long))
  {
    const UF_long* _Ap = reinterpret_cast<const UF_long*>(Ap);
    const UF_long* _Ai = reinterpret_cast<const UF_long*>(Ai);
    status = umfpack_dl_numeric(_Ap, _Ai, Ax, symbolic, &numeric, dnull.get(), dnull.get());
    umfpack_check_status(status, "numeric");
    return boost::shared_ptr<void>(numeric, UmfpackLongIntNumericDeleter());
  }
  else
    error("Could not determine correct types for casting integers to pass to UMFPACK.");

  return boost::shared_ptr<void>();
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::umfpack_solve(const std::size_t* Ap,
                                    const std::size_t* Ai,
                                    const double* Ax, double* x,
                                    const double* b, void* numeric)
{
  assert(Ap);
  assert(Ai);
  assert(Ax);
  assert(x);
  assert(b);
  assert(numeric);

  boost::scoped_ptr<double> dnull;

  // Solve system. We assume CSR storage, but UMFPACK expects CSC, so solve
  // for the transpose
  long int status = 0;
  if (sizeof(std::size_t) == sizeof(int))
  {
    const int* _Ap = reinterpret_cast<const int*>(Ap);
    const int* _Ai = reinterpret_cast<const int*>(Ai);
    status = umfpack_di_solve(UMFPACK_At, _Ap, _Ai, Ax, x, b, numeric, dnull.get(), dnull.get());
  }
  else if (sizeof(std::size_t) == sizeof(UF_long))
  {
    const UF_long* _Ap = reinterpret_cast<const UF_long*>(Ap);
    const UF_long* _Ai = reinterpret_cast<const UF_long*>(Ai);
    status = umfpack_dl_solve(UMFPACK_At, _Ap, _Ai, Ax, x, b, numeric, dnull.get(), dnull.get());
  }
  else
    error("Could not determine correct types for casting integers to pass to UMFPACK.");

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
boost::shared_ptr<void> UmfpackLUSolver::umfpack_factorize_symbolic(uint M, uint N,
                                                  const std::size_t* Ap,
                                                  const std::size_t* Ai,
                                                  const double* Ax)
{
  error("Umfpack not installed. Cannot perform LU solver using Umfpack.");
  return boost::shared_ptr<void>();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<void> UmfpackLUSolver::umfpack_factorize_numeric(const std::size_t* Ap,
                                                 const std::size_t* Ai,
                                                 const double* Ax,
                                                 void* symbolic)
{
  error("Umfpack not installed. Cannot perform LU solver using Umfpack.");
  return boost::shared_ptr<void>();
}
//-----------------------------------------------------------------------------
void UmfpackLUSolver::umfpack_solve(const std::size_t* Ap,
                                    const std::size_t* Ai,
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

