// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __NEW_MATRIX_H
#define __NEW_MATRIX_H

#include <petscmat.h>
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

    class Element;

    /// Constructor
    NewMatrix();

    /// Constructor
    NewMatrix(uint M, uint N);

    /// Constructor (just for testing, will be removed)
    NewMatrix(const Matrix &B);

    /// Destructor
    ~NewMatrix();

    /// Initialize matrix: no rows m, columns n, block size bs, 
    /// and max number of connectivity mnc. 
    void init(uint M, uint N);
    void init(uint M, uint N, uint bs);
    void init(uint M, uint N, uint bs, uint mnc);

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    uint size(uint dim) const;

    /// Set all entries to zero
    NewMatrix& operator= (real zero);

    /// Add block of values
    void add(const real block[], const int rows[], int m, const int cols[], int n);

    /// Set given rows to identity matrix
    void ident(const int rows[], int m);
    
    /// Matrix-vector multiplication
    void mult(const NewVector& x, NewVector& Ax) const;

    /// Apply changes to matrix
    void apply();

    /// Return PETSc Mat pointer
    Mat mat();

    /// Return PETSc Mat pointer
    const Mat mat() const;

    /// Display matrix (sparse output is default)
    void disp(bool sparse = true, int precision = 2) const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const NewMatrix& A);
    
    /// Element access operator (needed for const objects)
    real operator() (uint i, uint j) const;

    /// Element assignment operator
    Element operator()(uint i, uint j);

    class Element
    {
    public:
      Element(uint i, uint j, NewMatrix& A);
      operator real() const;
      const Element& operator=(const real a);
      const Element& operator=(const Element& e); 
      const Element& operator+=(const real a);
      const Element& operator-=(const real a);
      const Element& operator*=(const real a);
    protected:
      uint i, j;
      NewMatrix& A;
    };

  protected:

    // Element access
    real getval(uint i, uint j) const;

    // Set value of element
    void setval(uint i, uint j, const real a);
    
    // Add value to element
    void addval(uint i, uint j, const real a);
    
  private:

    // PETSc Mat pointer
    Mat A;

  };

}

#endif
