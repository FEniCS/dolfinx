// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-01
// Last changed: 2006-07-10

#include <dolfin/dolfin_log.h>
#include <dolfin/uBlasLUSolver.h>
#include <dolfin/uBlasKrylovSolver.h>

extern "C" 
{
// Take care of different default locations
#ifdef HAVE_UMFPACK_H
  #include <umfpack.h>
#elif HAVE_UMFPACK_UMFPACK_H
  #include <umfpack/umfpack.h>
#elif HAVE_UFSPARSE_UMFPACK_H
  #include <ufsparse/umfpack.h>
#endif
}

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasLUSolver::uBlasLUSolver() : LinearSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasLUSolver::~uBlasLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
#if defined(HAVE_UMFPACK_H)|| defined(HAVE_UMFPACK_UMFPACK_H) || defined(HAVE_UFSPARSE_UMFPACK_H)

dolfin::uint uBlasLUSolver::solve(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b)
{
  dolfin_info("Solving linear system of size %d x %d (UMFPACK LU solver).", 
      A.size(0), A.size(1));


  uint M = A.size(0);

  // Create column major sparse matrix correpsonding to A
  ublas_sparse_matrix_cmajor Test(A);

  double* null = (double *) NULL;
  void *Symbolic, *Numeric;
  const unsigned int* Ap = &(A.index1_data() [0]);
  const unsigned int* Ai = &(A.index2_data() [0]);
  const double* Ax = &(A.value_data() [0]);
  double* xx = &(x.data() [0]);
  const double* bb = &(b.data() [0]);

  umfpack_di_symbolic (M, M, (const int*) Ap, (const int*) Ai, Ax, &Symbolic, null, null);
  umfpack_di_numeric ((const int*) Ap, (const int*) Ai, Ax, Symbolic, &Numeric, null, null);
  umfpack_di_free_symbolic(&Symbolic);
  umfpack_di_solve(UMFPACK_A, (const int*) Ap, (const int*) Ai, Ax, xx, bb, Numeric, null, null);
  umfpack_di_free_numeric(&Numeric);

  return 1;
}

#else

dolfin::uint uBlasLUSolver::solve(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b)
{
  dolfin_warning("UMFPACK must be installed to peform a LU solve for uBlas matrices. A Krylov iterative solver will be used instead.");

  uBlasKrylovSolver solver;
  return solver.solve(A, x, b);
}
#endif
//-----------------------------------------------------------------------------
