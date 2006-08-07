// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed: 2006-07-10

#ifndef __UBLAS_LU_SOLVER_H
#define __UBLAS_LU_SOLVER_H

#include <dolfin/ublas.h>
#include <dolfin/Parametrized.h>
#include <dolfin/LinearSolver.h>

namespace dolfin
{

  /// Forward declarations
  class uBlasVector;
  template<class Mat> class uBlasMatrix;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b using uBlas data types.
  
  class uBlasLUSolver : public LinearSolver, public Parametrized
  {

  public:
    
    /// Constructor
    uBlasLUSolver();

    /// Destructor
    ~uBlasLUSolver();

    /// Solve linear system Ax = b (A is dense)
    uint solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, const uBlasVector& b) const;

    /// Solve linear system Ax = b in place (A is dense)
    uint solveInPlaceUBlas(uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, const uBlasVector& b) const;

    /// Compute the inverse of A (A is dense)
    void invert(uBlasMatrix<ublas_dense_matrix>& A) const;

    /// Compute the inverse of A (A is sparse)
    void invert(uBlasMatrix<ublas_sparse_matrix>& A) const
      {
      }

    /// Solve linear system Ax = b (A is sparse)
    /// UMFPACK is used if it has been configured. Otherwise a Krylov is used.
    uint solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, const uBlasVector& b);

  private:
    
    /// General uBlas LU solver which accepts both vector and matrix right-hand sides
    template<class Mat, class B>
    uint solveInPlace(Mat& A, B& z) const;

  };
  //---------------------------------------------------------------------------
  // Implementation of template functions
  //---------------------------------------------------------------------------
  template<class Mat, class B>
  dolfin::uint uBlasLUSolver::solveInPlace(Mat& A, B& X) const
  {
    const uint M = A.size1();
    dolfin_assert( M == A.size2() );
  
    // Create permutation matrix
    ublas::permutation_matrix<std::size_t> pmatrix(M);

    // Factorise (with pivoting)
    uint singular = ublas::lu_factorize(A, pmatrix);
    if( singular > 0)
      dolfin_error1("Singularity detected in uBlas matrix factorization on line %u.", singular-1); 

    // Back substitute 
    ublas::lu_substitute(A, pmatrix, X);

    return 1;
  }
//-----------------------------------------------------------------------------


}

#endif
