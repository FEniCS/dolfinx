// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
// 
// First added:  2006-06-01
// Last changed: 2006-08-08

#include <dolfin/dolfin_log.h>
#include <dolfin/uBlasLUSolver.h>
#include <dolfin/uBlasKrylovSolver.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/uBlasKrylovMatrix.h>
#include <dolfin/uBlasVector.h>

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
uBlasLUSolver::uBlasLUSolver() : uBlasLinearSolver(), AA(0), ej(0), Aj(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasLUSolver::~uBlasLUSolver()
{
  if ( AA ) delete AA;
  if ( ej ) delete ej;
  if ( Aj ) delete Aj;
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_dense_matrix>& A, 
                                  uBlasVector& x, const uBlasVector& b)
{    
  // Make copy of matrix and vector
  ublas_dense_matrix Atemp(A);
  x.resize(b.size());
  x.assign(b);

  // Solve
  return solveInPlace(Atemp, x);
}
//-----------------------------------------------------------------------------
#if defined(HAVE_UMFPACK_H)|| defined(HAVE_UMFPACK_UMFPACK_H) || defined(HAVE_UFSPARSE_UMFPACK_H)
dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, 
    const uBlasVector& b)
{
  // Check dimensions and get number of non-zeroes
  const uint M  = A.size(0);
  const uint N  = A.size(1);
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

dolfin::uint uBlasLUSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, 
    const uBlasVector& b)
{
  dolfin_warning("UMFPACK must be installed to peform a LU solve for uBlas matrices. A Krylov iterative solver will be used instead.");

  uBlasKrylovSolver solver;
  return solver.solve(A, x, b);
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

  // Reset unit vector
  *ej = 0.0;

  // Compute columns of matrix
  for (uint j = 0; j < N; j++)
  {
    (*ej)(j) = 1.0;

    // Compute product Aj = Aej
    A.mult(*ej, *Aj);
    
    // Set column of A
    column(*AA, j) = *Aj;
    
    (*ej)(j) = 0.0;
  }

  // Solve linear system
  solve(*AA, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::solveInPlaceUBlas(uBlasMatrix<ublas_dense_matrix>& A, 
                                      uBlasVector& x, const uBlasVector& b) const
{
  const uint M = A.size1();
  dolfin_assert(M == b.size());
  
  if( x.size() != M )
    x.resize(M);

  // Initialise solution vector
  x.assign(b);

  // Solve
  return solveInPlace(A, x);
}
//-----------------------------------------------------------------------------

