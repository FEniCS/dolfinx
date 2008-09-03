// Copyright (C) 2006-2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-05-15
// Last changed: 2008-05-17

#ifndef __MATRIX_H
#define __MATRIX_H

#include <dolfin/common/Variable.h>
#include "DefaultFactory.h"
#include "GenericMatrix.h"

namespace dolfin
{

  /// This class provides the default DOLFIN matrix class,
  /// based on the default DOLFIN linear algebra backend.

  class Matrix : public GenericMatrix, public Variable
  {
  public:

    /// Create empty matrix
    Matrix() : Variable("A", "DOLFIN matrix"), matrix(0)
    { DefaultFactory factory; matrix = factory.createMatrix(); }

    /// Create M x N matrix
    Matrix(uint M, uint N) : Variable("A", "DOLFIN matrix"), matrix(0)
    { DefaultFactory factory; matrix = factory.createMatrix(); matrix->init(M, N); }

    /// Copy constructor
    explicit Matrix(const Matrix& A) : Variable("A", "DOLFIN matrix"),
                                       matrix(A.matrix->copy())
    {}

    /// Destructor
    virtual ~Matrix()
    { delete matrix; }

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern)
    { matrix->init(sparsity_pattern); }

    /// Return copy of tensor
    virtual Matrix* copy() const
    { Matrix* A = new Matrix(); delete A->matrix; A->matrix = matrix->copy(); return A; }

    /// Return size of given dimension
    virtual uint size(uint dim) const
    { return matrix->size(dim); }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    { matrix->zero(); }

    /// Finalize assembly of tensor
    virtual void apply()
    { matrix->apply(); }

    /// Display tensor
    virtual void disp(uint precision=2) const
    { matrix->disp(precision); }

    //--- Implementation of the GenericMatrix interface ---

    /// Initialize M x N matrix
    virtual void init(uint M, uint N)
    { matrix->init(M, N); }

    /// Get block of values
    virtual void get(real* block, uint m, const uint* rows, uint n, const uint* cols) const
    { matrix->get(block, m, rows, n, cols); }

    /// Set block of values
    virtual void set(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    { matrix->set(block, m, rows, n, cols); }

    /// Add block of values
    virtual void add(const real* block, uint m, const uint* rows, uint n, const uint* cols)
    { matrix->add(block, m, rows, n, cols); }

    /// Get non-zero values of given row
    virtual void getrow(uint row, Array<uint>& columns, Array<real>& values) const
    { matrix->getrow(row, columns, values); }

    /// Set values for given row
    virtual void setrow(uint row, const Array<uint>& columns, const Array<real>& values)
    { matrix->setrow(row, columns, values); }

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows)
    { matrix->zero(m, rows); }

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows)
    { matrix->ident(m, rows); }

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y, bool transposed=false) const
    { matrix->mult(x, y, transposed); }

    /// Multiply matrix by given number
    virtual const Matrix& operator*= (real a)
    { *matrix *= a; return *this; }

    /// Divide matrix by given number
    virtual const Matrix& operator/= (real a)
    { *matrix /= a; return *this; }

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A)
    { *matrix = A; return *this; }

    /// Return pointers to underlying compressed storage data
    virtual boost::tuple<const std::size_t*, const std::size_t*, const double*, int> data() const
    { return matrix->data(); }

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const
    { return matrix->factory(); }

    //--- Special functions, intended for library use only ---

    /// Return concrete instance / unwrap (const version)
    virtual const GenericMatrix* instance() const
    { return matrix; }

    /// Return concrete instance / unwrap (non-const version)
    virtual GenericMatrix* instance()
    { return matrix; }

    //--- Special Matrix functions ---

    /// Assignment operator
    const Matrix& operator= (const Matrix& A)
    { *matrix = *A.matrix; return *this; }

  private:

    // Pointer to concrete implementation
    GenericMatrix* matrix;

  };

}

#endif
