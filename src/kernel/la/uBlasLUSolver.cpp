// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-01
// Last changed: 2006-07-10

#include <dolfin/timing.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/uBlasLUSolver.h>
#include <dolfin/uBlasKrylovSolver.h>
#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasSparseMatrix.h>

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
dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_dense_matrix>& A, 
                                  DenseVector& x, const DenseVector& b) const
{    
  // Make copy of matrix and vector
  ublas_dense_matrix Atemp(A);
  x.resize(b.size());
  x.assign(b);

  // Solve
  solveInPlace(Atemp, x);

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::solveInPlaceUBlas(uBlasMatrix<ublas_dense_matrix>& A, 
                                      DenseVector& x, const DenseVector& b) const
{
  const uint M = A.size1();
  dolfin_assert(M == b.size());
  
  if( x.size() != M )
    x.resize(M);

  // Initialise solution vector
  x.assign(b);

  // Solve
  solveInPlace(A, x);

  return 1;
}
//-----------------------------------------------------------------------------
void uBlasLUSolver::invert(uBlasMatrix<ublas_dense_matrix>& A) const
{
  const uint M = A.size1();
  dolfin_assert(M == A.size2());
  
  // Create indentity matrix
  ublas_dense_matrix X(M, M);
  X.assign(ublas::identity_matrix<real>(M));

  // Solve
  solveInPlace(A, X);

  A.assign_temporary(X);
}
//-----------------------------------------------------------------------------
#if defined(HAVE_UMFPACK_H)|| defined(HAVE_UMFPACK_UMFPACK_H) || defined(HAVE_UFSPARSE_UMFPACK_H)

dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, DenseVector& x, 
    const DenseVector& b)
{
  // Check dimensions and get number of non-zeroes
  const uint M = A.size(0);
  const uint N = A.size(1);
  const uint nz = A.nnz();

  dolfin_assert(M == A.size(1));
  dolfin_assert(nz > 0);

  x.init(N);

  dolfin_info("Solving linear system of size %d x %d (UMFPACK LU solver).", 
      M, N);

  double* dnull = (double *) NULL;
  int*    inull = (int *) NULL;
  void *Symbolic, *Numeric;
  const unsigned int* Ap = &(A.index1_data() [0]);
  const unsigned int* Ai = &(A.index2_data() [0]);
  const double* Ax = &(A.value_data() [0]);
  double* xx = &(x.data() [0]);
  const double* bb = &(b.data() [0]);


  // Solve for transpose since we use compressed row format, and UMFPACK 
  // expects compressed column format

  int* Rp = new int[M+1];
  int* Ri = new int[nz];
  double* Rx = new double[nz];

  // Compute transpose
  umfpack_di_transpose(M, M, (const int*) Ap, (const int*) Ai, Ax, inull, inull, Rp, Ri, Rx);

  // Solve procedure
  umfpack_di_symbolic(M, M, (const int*) Rp, (const int*) Ri, Rx, &Symbolic, dnull, dnull);
  umfpack_di_numeric( (const int*) Rp, (const int*) Ri, Rx, Symbolic, &Numeric, dnull, dnull);
  umfpack_di_free_symbolic(&Symbolic);
  umfpack_di_solve(UMFPACK_A, (const int*) Rp, (const int*) Ri, Rx, xx, bb, Numeric, dnull, dnull);
  umfpack_di_free_numeric(&Numeric);

  // Clean up
  delete [] Rp;
  delete [] Ri;
  delete [] Rx;

  return 1;
}

#else

dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, DenseVector& x, 
    const DenseVector& b)
{
  dolfin_warning("UMFPACK must be installed to peform a LU solve for uBlas matrices. A Krylov iterative solver will be used instead.");

  uBlasKrylovSolver solver;
  return solver.solve(A, x, b);
}
#endif
//-----------------------------------------------------------------------------

