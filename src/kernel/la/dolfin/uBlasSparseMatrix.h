// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-05-29
// Last changed: 2006-06-29

#ifndef __UBLAS_SPARSE_MATRIX_H
#define __UBLAS_SPARSE_MATRIX_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/Variable.h>
#include <dolfin/GenericMatrix.h>
#include <dolfin/uBlasKrylovMatrix.h>
#include <dolfin/ublas.h>

namespace dolfin
{

  class DenseVector;
  namespace ublas = boost::numeric::ublas;

  /// This class represents a sparse matrix of dimension M x N.
  /// It is a simple wrapper for a Boost ublas sparse matrix.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// refer to the documentation for ublas which can be found at
  /// http://www.boost.org/libs/numeric/ublas/doc/index.htm.

  class uBlasSparseMatrix : public Variable,
			    public GenericMatrix,
			    public uBlasKrylovMatrix,
			    public ublas_sparse_matrix
  {
  public:
    
    /// Constructor
    uBlasSparseMatrix();
    
    /// Constructor
    uBlasSparseMatrix(uint M, uint N);

    /// Destructor
    ~uBlasSparseMatrix();

    /// Initialize M x N matrix
    void init(const uint M, const uint N);

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    /// nzmax is a dummy argument ans is passed for compatibility with sparse matrces
    void init(const uint M, const uint N, const uint nzmax);

    /// Assignment from a matrix_expression
    template <class E>
    uBlasSparseMatrix& operator=(const ublas::matrix_expression<E>& A) const
    { 
      ublas_sparse_matrix::operator=(A); 
      return *this;
    } 

    /// Return number of rows (dim = 0) or columns (dim = 1) 
    uint size(const uint dim) const;

    /// Set block of values. The function apply() must be called to commit changes.
    void set(const real block[], const int rows[], int m, const int cols[], const int n);

    /// Add block of values. The function apply() must be called to commit changes.
    void add(const real block[], const int rows[], int m, const int cols[], const int n);

    /// Get non-zero values of row i
    void getRow(const uint i, int& ncols, Array<int>& columns, Array<real>& values) const;

    /// Lump matrix into vector m
    void lump(DenseVector& m) const;

    /// Solve Ax = b out-of-place (A is not destroyed)
    void solve(DenseVector& x, const DenseVector& b) const;

    /// Solve Ax = b in-place (A is destroyed)
    void solveInPlace(DenseVector& x, const DenseVector& b);

    /// Apply changes to matrix 
    void apply();

    /// Set all entries to zero
    void zero();

    /// Set given rows to identity matrix
    void ident(const int rows[], const int m);

    /// Average number of non-zero terms per row
    uint nzmax() const
     { return nnz()/size(0); }

    /// Compute A*x
    void mult(const DenseVector& x, DenseVector& Ax) const;

    /// Display matrix
    void disp(const uint precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const uBlasSparseMatrix& A);

   private:
  
    /// Matrix used internally for assembly
    ublas_assembly_matrix Assembly_matrix;

    /// Matrix state
    bool assembled;

  };
}

#endif
