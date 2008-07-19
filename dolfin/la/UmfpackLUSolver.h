// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Dag Lindbo 2008.
// 
// First added:  2006-05-31
// Last changed: 2008-07-19

#ifndef __UMFPACK_LU_SOLVER_H
#define __UMFPACK_LU_SOLVER_H

#include "ublas.h"
#include <dolfin/parameter/Parametrized.h>

namespace dolfin
{

  /// Forward declarations
  class uBlasVector;
  class uBlasKrylovMatrix;
  template<class Mat> class uBlasMatrix;

  /// This class implements the direct solution (LU factorization) of
  /// linear systems of the form Ax = b using uBlas data types. Dense
  /// matrices are solved using uBlas LU factorisation, and sparse matrices
  /// are solved using UMFPACK (http://www.cise.ufl.edu/research/sparse/umfpack/)
  /// is installed. Matrices can also be inverted.
    
  class UmfpackLUSolver : public Parametrized
  {

  public:
    
    /// Constructor
    UmfpackLUSolver();

    /// Destructor
    ~UmfpackLUSolver();

    /// Solve linear system Ax = b for a dense matrix
    virtual uint solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, const uBlasVector& b);

    /// Solve linear system Ax = b for a sparse matrix using UMFPACK if installed
    virtual uint solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, const uBlasVector& b);

    /// LU-factor sparse matrix A if UMFPACK is installed
    virtual uint factorize(const uBlasMatrix<ublas_sparse_matrix>& A);

    /// Solve factorized system (UMFPACK).
    virtual uint factorizedSolve(uBlasVector& x, const uBlasVector& b);

    /// Solve linear system Ax = b for a Krylov matrix
    void solve(const uBlasKrylovMatrix& A, uBlasVector& x, const uBlasVector& b);

    /// Solve linear system Ax = b in place (A is dense)
    uint solveInPlaceUBlas(uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, const uBlasVector& b) const;

    /// Compute the inverse of A (A is dense or sparse)
    template<class Mat>
    void invert(Mat& A) const;

  private:
    
    /// General uBlas LU solver which accepts both vector and matrix right-hand sides
    template<class Mat, class B>
    uint solveInPlace(Mat& A, B& X) const;

    // Temporary data for LU factorization of sparse ublas matrix (umfpack only)
    class Umfpack
    {
      public:
 
        Umfpack() : dnull(0), inull(0), Symbolic(0), Numeric(0), local_matrix(0), Rp(0), Ri(0), Rx(0), 
                     N(0), factorized(false), mat_dim(0) {} 

        ~Umfpack() { clear(); }

        // Clear data
        void clear();

        // Initialise with matrix
        void init(const long int* Ap, const long int* Ai, const double* Ax, uint M, uint nz);

        // Initialise with transpose of matrix
        void initTranspose(const long int* Ap, const long int* Ai, const double* Ax, uint M, uint nz);

        // Factorize
        void factorize();

        // Factorized solve
        void factorizedSolve(double*x, const double* b, bool transpose = false);

        /// Check status flag returned by an UMFPACK function
        void checkStatus(long int status, std::string function);

        // UMFPACK data
        double*   dnull;
        long int* inull;
        void *Symbolic;
        void* Numeric;

        // Matrix data
        bool local_matrix;
        const long int* Rp;
        const long int* Ri;
        const double*   Rx;   

        uint N;
        bool factorized;
        uint mat_dim;
    };

    Umfpack umfpack;

    // Temporary data for LU factorization of a uBlasKrylovMatrix
    uBlasMatrix<ublas_dense_matrix>* AA;
    uBlasVector* ej;
    uBlasVector* Aj;
    
  };
  //---------------------------------------------------------------------------
  // Implementation of template functions
  //---------------------------------------------------------------------------
  template<class Mat>
  void UmfpackLUSolver::invert(Mat& A) const
  {
    const uint M = A.size1();
    dolfin_assert(M == A.size2());
  
    // Create indentity matrix
    Mat X(M, M);
    X.assign(ublas::identity_matrix<real>(M));

    // Solve
    solveInPlace(A, X);

    A.assign_temporary(X);
  }
  //-----------------------------------------------------------------------------
  template<class Mat, class B>
  dolfin::uint UmfpackLUSolver::solveInPlace(Mat& A, B& X) const
  {
    const uint M = A.size1();
    dolfin_assert( M == A.size2() );
  
    // Create permutation matrix
    ublas::permutation_matrix<std::size_t> pmatrix(M);

    // Factorise (with pivoting)
    uint singular = ublas::lu_factorize(A, pmatrix);
    if( singular > 0)
      error("Singularity detected in uBlas matrix factorization on line %u.", singular-1); 

    // Back substitute 
    ublas::lu_substitute(A, pmatrix, X);

    return 1;
  }
  //-----------------------------------------------------------------------------

}

#endif
