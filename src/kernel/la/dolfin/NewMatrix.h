// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_MATRIX_H
#define __NEW_MATRIX_H

#include <petsc/petscmat.h>
#include <dolfin/constants.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/Matrix.h>

namespace dolfin
{
  
  /// This class represents a matrix of dimension m x n. It is a
  /// simple wrapper for a PETSc matrix (Mat). The interface is
  /// intentionally simple. For advanced usage, access the PETSc Mat
  /// pointer using the function mat() and use the standard PETSc
  /// interface.

  class NewVector;

  class NewMatrix
  {
  public:

    class Index;

    /// Constructor
    NewMatrix();

    /// Constructor
    NewMatrix(int m, int n);

    /// Constructor
    NewMatrix(const Matrix &B);

    /// Destructor
    ~NewMatrix();

    /// Initialize matrix: no rows m, columns n, block size bs, 
    /// and max number of connectivity mnc. 
    void init(int m, int n);
    void init(int m, int n, int bs);
    void init(int m, int n, int bs, int mnc);

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension
    /// dim
    int size(int dim) const;

    /// Set all entries to zero
    NewMatrix& operator= (real zero);

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n);

    /// Element assignment
    void setvalue(int i, int j, const real r);

    /// Element access
    real getvalue(int i, int j) const;

    /// Matrix-vector multiplication
    void mult(const NewVector& x, NewVector& Ax) const;

    /// Apply changes to matrix
    void apply();

    /// Return PETSc Mat pointer
    Mat mat();

    /// Return PETSc Mat pointer
    const Mat mat() const;

    /// Display matrix
    void disp() const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const NewMatrix& A);
    
    /// Element assignment operator
    Index operator()(int i, int j);

    class Index
    {
    public:
      Index(int i, int j, NewMatrix &m);

      operator real() const;
      void operator=(const real r);

    protected:
      int i, j;
      NewMatrix &m;
    };


  private:

    // PETSc Mat pointer
    Mat A;

  };

}

#endif
