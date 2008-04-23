// Copyright (C) 2006-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Kent-Andre Mardal 2008.
//
// First added:  2006-05-15
// Last changed: 2008-04-22

#ifndef __MATRIX_H
#define __MATRIX_H

#include "default_la_types.h"
#include "GenericMatrix.h"

namespace dolfin
{
  
  /// This class provides an interface to the default DOLFIN
  /// matrix implementation as decided in default_la_types.h.
  
  class Matrix : public GenericMatrix, public Variable
  {
  public:
    
    /// Constructor
    Matrix() : GenericMatrix(), Variable("A", "DOLFIN matrix"),
	       matrix(new DefaultMatrix()) {}
    
    /// Constructor
    Matrix(uint M, uint N) : GenericMatrix(), Variable("A", "DOLFIN matrix"),
			     matrix(new DefaultMatrix(M, N)) {}
    
    /// Destructor
    ~Matrix()
     { delete matrix; }
    
    /// Initialize M x N matrix
    void init(uint M, uint N)
    { matrix->init(M, N); }
    
    /// Initialize zero matrix using sparsity pattern
    void init(const GenericSparsityPattern& sparsity_pattern)
    { matrix->init(sparsity_pattern); }
    
    /// Return copy of matrix
    Matrix* copy() const
    { Matrix* A = new Matrix(); delete A->matrix; A->matrix = matrix->copy(); return A; }

    /// Return size of given dimension
    uint size(uint dim) const
    { return matrix->size(dim); }

    /// Get block of values
    void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const
    { matrix->get(block, m, rows, n, cols); }
    
    /// Set block of values
    void set(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    { matrix->set(block, m, rows, n, cols); }
    
    /// Add block of values
    void add(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    { matrix->add(block, m, rows, n, cols); }

    /// Set all entries to zero and keep any sparse structure (implemented by sub class)
    void zero()
    { matrix->zero(); }

    /// Set given rows to zero matrix
    void zero(uint m, const uint* rows)
    { matrix->zero(m, rows); }
    
    /// Set given rows to identity matrix
    void ident(uint m, const uint* rows)
    { matrix->ident(m, rows); }
        
    /// Finalise assembly of matrix
    void apply()
    { matrix->apply(); }
    
    /// Display matrix (sparse output is default)
    void disp(uint precision = 2) const
    { matrix->disp(precision); }

    /// Multiply matrix by given number
    virtual const Matrix& operator*= (real a)
    { *matrix *= a; return *this; }

    /// Get non-zero values of row i
    void getrow(uint i, int& ncols, Array<int>& columns, Array<real>& values) const
    { matrix->getrow(i, ncols, columns, values); }
    
    LinearAlgebraFactory& factory() const
    { return matrix->factory(); }

    // y = A x  ( or y = A^T x if transposed==true) 
    void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const 
    { matrix->mult(x, y, transposed); }

    /// Assignment operator
    const GenericMatrix& operator= (const GenericMatrix& A)
    { *matrix = A; return *this; }

    /// Assignment operator
    const Matrix& operator= (const Matrix& A)
    { *matrix = *A.matrix; return *this; }

    ///--- Special functions, intended for library use only ---

    /// Return instance (const version)
    virtual const GenericMatrix* instance() const 
    { return matrix; }

    /// Return instance (non-const version)
    virtual GenericMatrix* instance() 
    { return matrix; }

  private:

    GenericMatrix* matrix;
    
  };

}

#endif
