// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Dag Lindbo 2008.
// 
// First added:  2006-05-31
// Last changed: 2008-07-21

#ifndef __UMFPACK_LU_SOLVER_H
#define __UMFPACK_LU_SOLVER_H

#include "ublas.h"
#include <dolfin/parameter/Parametrized.h>

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  class GenericMatrix;
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

    /// Solve uBLAS linear system Ax = b for a sparse matrix using UMFPACK if installed
    virtual uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// LU-factor sparse matrix A if UMFPACK is installed
    virtual uint factorize(const GenericMatrix& A);

    /// Solve factorized system (UMFPACK).
    virtual uint factorizedSolve(GenericVector& x, const GenericVector& b) const;

    /// Solve linear system Ax = b for a Krylov matrix
    /// FIXME: This function should be moved to uBlasKrylovMatrix
    void solve(const uBlasKrylovMatrix& A, uBlasVector& x, const uBlasVector& b);

  private:
    
    // Data for LU factorization of sparse ublas matrix (umfpack only)
    class Umfpack
    {
      public:
 
        Umfpack() : dnull(0), inull(0), Symbolic(0), Numeric(0), local_matrix(0), 
                    Rp(0), Ri(0), Rx(0), N(0), factorized(false), mat_dim(0) {} 

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
        void factorizedSolve(double*x, const double* b, bool transpose = false) const;

        /// Check status flag returned by an UMFPACK function
        void checkStatus(long int status, std::string function) const;

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

}

#endif
