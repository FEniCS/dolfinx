// Copyright (C) 2004-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2006.
//
// First added:  2004
// Last changed: 2006-08-14

#ifndef __PETSC_MATRIX_H
#define __PETSC_MATRIX_H

#ifdef HAVE_PETSC_H

#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/Variable.h>
#include <dolfin/GenericMatrix.h>

namespace dolfin
{
  
  /// Forward declarations
  class PETScVector;
  class PETScMatrixElement;
  
  template<class M>
  class Array;

  /// This class represents a sparse matrix of dimension M x N.
  /// It is a simple wrapper for a PETSc matrix pointer (Mat).
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Mat pointer using the function mat() and
  /// use the standard PETSc interface.

  class PETScMatrix : public GenericMatrix, public Variable
  {
  public:

    /// PETSc sparse matrix types
    enum Type
    { 
      default_matrix, // Default matrix type 
      spooles,        // Spooles
      superlu,        // Super LU
      umfpack         // UMFPACK
    };

    /// Constructor
    PETScMatrix();

    /// Constructor (setting PETSc matrix type)
    PETScMatrix(Type type);

    /// Constructor
    PETScMatrix(Mat A);

    /// Constructor
    PETScMatrix(uint M, uint N);

    /// Constructor (setting PETSc matrix type)
    PETScMatrix(uint M, uint N, Type type);

    /// Constructor (just for testing, will be removed)
    PETScMatrix(const PETScMatrix &B);

    /// Destructor
    ~PETScMatrix();

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
   
    /// Access element value
    real get(uint i, uint j) const;

    /// Set element value
    void set(uint i, uint j, real value);

    /// Set block of values
    void set(const real block[], const int rows[], int m, const int cols[], int n);

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n);

    /// Get non-zero values of row i
    void getRow(const uint i, int& ncols, Array<int>& columns, Array<real>& values) const;

    /// Set given rows to identity matrix
    void ident(const int rows[], int m);
    
    /// Matrix-vector multiplication
    void mult(const PETScVector& x, PETScVector& Ax) const;

    /// Matrix-vector multiplication with given row (temporary fix, assumes uniprocessor case)
    real mult(const PETScVector& x, uint row) const;

    /// Matrix-vector multiplication with given row (temporary fix, assumes uniprocessor case)
    real mult(const real x[], uint row) const;

    /// Lump matrix into vector m
    void lump(PETScVector& m) const;

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
    friend LogStream& operator<< (LogStream& stream, const PETScMatrix& A);
    
    /// SparseMatrixElement access operator (needed for const objects)
    real operator() (uint i, uint j) const;

    /// SparseMatrixElement assignment operator
    PETScMatrixElement operator()(uint i, uint j);

    // Friends
    friend class PETScMatrixElement;

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

  class PETScMatrixElement
  {
  public:
    PETScMatrixElement(uint i, uint j, PETScMatrix& A);
    PETScMatrixElement(const PETScMatrixElement& e);
    operator real() const;
    const PETScMatrixElement& operator=(const real a);
    const PETScMatrixElement& operator=(const PETScMatrixElement& e); 
    const PETScMatrixElement& operator+=(const real a);
    const PETScMatrixElement& operator-=(const real a);
    const PETScMatrixElement& operator*=(const real a);
  protected:
    uint i, j;
    PETScMatrix& A;
  };

}

#endif

#endif
