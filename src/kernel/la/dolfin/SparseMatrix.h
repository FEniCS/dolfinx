// Copyright (C) 2004-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2006.
//
// First added:  2004
// Last changed: 2006-05-15

#ifndef __SPARSE_MATRIX_H
#define __SPARSE_MATRIX_H

#ifdef HAVE_PETSC_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/PETScManager.h>
#include <dolfin/GenericMatrix.h>

// FIXME: Rename to SparseMatrix
// FIXME: Make typedef
// FIXME: implement lump()

namespace dolfin
{
  
  class SparseVector;
  class SparseMatrixElement;

  /// This class represents a sparse matrix of dimension M x N.
  /// It is a simple wrapper for a PETSc matrix pointer (Mat).
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Mat pointer using the function mat() and
  /// use the standard PETSc interface.

  class SparseMatrix : public GenericMatrix, public Variable
  {
  public:

    /// Sparse matrix types
    enum Type
    { 
      default_matrix, // Default matrix type 
      spooles,        // Spooles
      superlu,        // Super LU
      umfpack         // UMFPACK
    };

    /// Constructor
    SparseMatrix();

    /// Constructor (setting PETSc matrix type)
    SparseMatrix(Type type);

    /// Constructor
    SparseMatrix(Mat A);

    /// Constructor
    SparseMatrix(uint M, uint N);

    /// Constructor (setting PETSc matrix type)
    SparseMatrix(uint M, uint N, Type type);

    /// Constructor (just for testing, will be removed)
    SparseMatrix(const SparseMatrix &B);

    /// Destructor
    ~SparseMatrix();

    /// Initialize M x N matrix
    void init(uint M, uint N);

    /// Initialize M x N matrix with given maximum number of nonzeros in each row
    void init(uint M, uint N, uint nzmax);

    /// Initialize M x N matrix with given block size and maximum number of nonzeros in each row
    void init(uint M, uint N, uint bs, uint nzmax);

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    uint size(uint dim) const;

    /// Return number of nonzero entries in given row
    uint nz(uint row) const;

    /// Return total number of nonzero entries
    uint nzsum() const;

    /// Return maximum number of nonzero entries
    uint nzmax() const;
    
    /// Set block of values
    void set(const real block[], const int rows[], int m, const int cols[], int n);

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n);

    /// Set given rows to identity matrix
    void ident(const int rows[], int m);
    
    /// Matrix-vector multiplication
    void mult(const SparseVector& x, SparseVector& Ax) const;

    /// Matrix-vector multiplication with given row (temporary fix, assumes uniprocessor case)
    real mult(const SparseVector& x, uint row) const;

    /// Matrix-vector multiplication with given row (temporary fix, assumes uniprocessor case)
    real mult(const real x[], uint row) const;

    /// Compute given norm of matrix
    enum Norm { l1, linf, frobenius };
    real norm(Norm type = l1) const;

    /// Apply changes to matrix
    void apply();

    /// Set all entries to zero
    void zero();

    /// Return matrix type 
    Type type() const;

    /// Return PETSc Mat pointer
    Mat mat() const;

    /// Display matrix (sparse output is default)
    void disp(bool sparse = true, int precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const SparseMatrix& A);
    
    /// SparseMatrixElement access operator (needed for const objects)
    real operator() (uint i, uint j) const;

    /// SparseMatrixElement assignment operator
    SparseMatrixElement operator()(uint i, uint j);

    // Friends
    friend class SparseMatrixElement;

    // MatrixElement access
    real getval(uint i, uint j) const;

    // Set value of element
    void setval(uint i, uint j, const real a);
    
    // Add value to element
    void addval(uint i, uint j, const real a);
    
  private:

    // PETSc Mat pointer
    Mat A;

    // PETSc matrix type
    Type _type;

    // Set matrix type 
    void setType();

    // Check that requested type has been compiled into PETSc
    void checkType();

    // Return PETSc matrix type 
    MatType getPETScType() const;

  };

  /// Reference to an element of the matrix

  class SparseMatrixElement
  {
  public:
    SparseMatrixElement(uint i, uint j, SparseMatrix& A);
    SparseMatrixElement(const SparseMatrixElement& e);
    operator real() const;
    const SparseMatrixElement& operator=(const real a);
    const SparseMatrixElement& operator=(const SparseMatrixElement& e); 
    const SparseMatrixElement& operator+=(const real a);
    const SparseMatrixElement& operator-=(const real a);
    const SparseMatrixElement& operator*=(const real a);
  protected:
    uint i, j;
    SparseMatrix& A;
  };

}

#endif

#endif
