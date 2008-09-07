// Copyright (C) 2006-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
// Modified by Dag Lindbo 2008.
// 
// First added:  2006-05-31
// Last changed: 2008-09-05

#ifndef __UMFPACK_LU_SOLVER_H
#define __UMFPACK_LU_SOLVER_H

#include "ublas.h"
#include "GenericLinearSolver.h"

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  class GenericMatrix;
  class uBLASVector;
  class uBLASKrylovMatrix;
  template<class Mat> class uBLASMatrix;

  /// This class implements the direct solution (LU factorization) of
  /// linear systems of the form Ax = b using uBLAS data types. Dense
  /// matrices are solved using uBLAS LU factorisation, and sparse matrices
  /// are solved using UMFPACK (http://www.cise.ufl.edu/research/sparse/umfpack/)
  /// is installed. Matrices can also be inverted.
    
  class UmfpackLUSolver : public GenericLinearSolver
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

  private:

#ifdef HAS_UMFPACK
    /// Data for LU factorization of sparse ublas matrix (umfpack only)
    class Umfpack
    {
      public:
 
        // Constructor
        Umfpack() : dnull(0), inull(0), Symbolic(0), Numeric(0), local_matrix(false), 
                    Rp(0), Ri(0), Rx(0), N(0), factorized(false) {} 

        // Destructor
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
    };

    Umfpack umfpack;
#endif
    
  };

}

#endif
