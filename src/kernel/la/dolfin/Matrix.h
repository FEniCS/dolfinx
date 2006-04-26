// Copyright (C) 2004-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy R. Terrel, 2005.
// Modified by Garth N. Wells, 2006.
//
// First added:  2004
// Last changed: 2006-04-25

#ifndef __MATRIX_H
#define __MATRIX_H

#include <petscmat.h>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/GenericMatrix.h>

namespace dolfin
{
  
  /// This class represents a matrix of dimension m x n. It is a
  /// simple wrapper for a PETSc matrix (Mat). The interface is
  /// intentionally simple. For advanced usage, access the PETSc Mat
  /// pointer using the function mat() and use the standard PETSc
  /// interface.

  class Vector;
  class MatrixElement;

  class Matrix : public GenericMatrix<Matrix>, public Variable
  {
  public:

    /// Matrix types
    enum Type
    { 
      default_matrix, // Default matrix type 
      spooles,        // Spooles
      superlu,        // Super LU
      umfpack         // UMFPACK
    };

    /// Constructor
    Matrix();

    /// Constructor (setting PETSc matrix type)
    Matrix(Type type);

    /// Constructor
    Matrix(Mat A);

    /// Constructor
    Matrix(uint M, uint N);

    /// Constructor (setting PETSc matrix type)
    Matrix(uint M, uint N, Type type);

    /// Constructor (just for testing, will be removed)
    Matrix(const Matrix &B);

    /// Destructor
    ~Matrix();

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
    
    /// Set all entries to zero
    Matrix& operator= (real zero);

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n);

    /// Set given rows to identity matrix
    void ident(const int rows[], int m);
    
    /// Matrix-vector multiplication
    void mult(const Vector& x, Vector& Ax) const;

    /// Matrix-vector multiplication with given row (temporary fix, assumes uniprocessor case)
    real mult(const Vector& x, uint row) const;

    /// Matrix-vector multiplication with given row (temporary fix, assumes uniprocessor case)
    real mult(const real x[], uint row) const;

    /// Compute given norm of matrix
    enum Norm { l1, linf, frobenius };
    real norm(Norm type = l1) const;

    /// Apply changes to matrix
    void apply();

    /// Return matrix type 
    Type getMatrixType() const;

    /// Return PETSc Mat pointer
    Mat mat() const;

    /// Display matrix (sparse output is default)
    void disp(bool sparse = true, int precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const Matrix& A);
    
    /// MatrixElement access operator (needed for const objects)
    real operator() (uint i, uint j) const;

    /// MatrixElement assignment operator
    MatrixElement operator()(uint i, uint j);

    // Friends
    friend class MatrixElement;

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
    Type type;

    // Set matrix type 
    void setType();

    // Check that requested type has been compiled into PETSc
    void checkType();

    // Return PETSc matrix type 
    MatType getType() const;

  };

  /// Reference to an element of the vector

  class MatrixElement
    {
    public:
      MatrixElement(uint i, uint j, Matrix& A);
      MatrixElement(const MatrixElement& e);
      operator real() const;
      const MatrixElement& operator=(const real a);
      const MatrixElement& operator=(const MatrixElement& e); 
      const MatrixElement& operator+=(const real a);
      const MatrixElement& operator-=(const real a);
      const MatrixElement& operator*=(const real a);
    protected:
      uint i, j;
      Matrix& A;
    };

}

#endif
