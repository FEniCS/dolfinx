// Copyright (C) 2006-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Jansson, 2006.
// Modified by Anders Logg, 2006-2008.
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Martin Alnæs, 2008.
//
// First added:  2006-04-24
// Last changed: 2010-02-24

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <tr1/tuple>
#include <vector>
#include "GenericTensor.h"

namespace dolfin
{

  class GenericVector;
  class XMLMatrix;

  /// This class defines a common interface for matrices.

  class GenericMatrix : public GenericTensor
  {
  public:

    /// Destructor
    virtual ~GenericMatrix() {}

    //--- Implementation of the GenericTensor interface ---

    /// Resize tensor with given dimensions
    virtual void resize(uint rank, const uint* dims)
    { assert(rank == 2); resize(dims[0], dims[1]); }

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern) = 0;

    /// Return copy of tensor
    virtual GenericMatrix* copy() const = 0;

    /// Return tensor rank (number of dimensions)
    virtual uint rank() const
    { return 2; }

    /// Return size of given dimension
    virtual uint size(uint dim) const = 0;

    /// Get block of values
    virtual void get(double* block, const uint* num_rows,
                     const uint * const * rows) const
    { get(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Set block of values
    virtual void set(const double* block, const uint* num_rows,
                     const uint * const * rows)
    { set(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Add block of values
    virtual void add(const double* block, const uint* num_rows,
                     const uint * const * rows)
    { add(block, num_rows[0], rows[0], num_rows[1], rows[1]); }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero() = 0;

    /// Finalize assembly of tensor
    virtual void apply(std::string mode) = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

    //--- Matrix interface ---

    /// Resize matrix to  M x N
    virtual void resize(uint M, uint N) = 0;

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows, uint n,
                     const uint* cols) const = 0;

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows, uint n,
                     const uint* cols) = 0;

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows, uint n,
                     const uint* cols) = 0;

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A,
                      bool same_nonzero_pattern) = 0;

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const = 0;

    /// Get non-zero values of given row on local process
    virtual void getrow(uint row, std::vector<uint>& columns,
                        std::vector<double>& values) const = 0;

    /// Set values for given row on local process
    virtual void setrow(uint row, const std::vector<uint>& columns,
                        const std::vector<double>& values) = 0;

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows) = 0;

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows) = 0;

    /// Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;

    /// Matrix-vector product, y = A^T x
    virtual void transpmult(const GenericVector& x, GenericVector& y) const = 0;

    /// Multiply matrix by given number
    virtual const GenericMatrix& operator*= (double a) = 0;

    /// Divide matrix by given number
    virtual const GenericMatrix& operator/= (double a) = 0;

    /// Add given matrix
    const GenericMatrix& operator+= (const GenericMatrix& A)
    {
      axpy(1.0, A, false);
      return *this;
    }

    /// Subtract given matrix
    const GenericMatrix& operator-= (const GenericMatrix& A)
    {
      axpy(-1.0, A, false);
      return *this;
    }

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& x) = 0;

    /// Return pointers to underlying compresssed row/column storage data
    /// For compressed row storage, data = (row_pointer[#rows +1],
    /// column_index[#nz], matrix_values[#nz], nz)
    virtual std::tr1::tuple<const std::size_t*, const std::size_t*,
                                            const double*, int> data() const
    {
      error("Unable to return pointers to underlying matrix data.");
      return std::tr1::tuple<const std::size_t*, const std::size_t*,
                                               const double*, int>(0, 0, 0, 0);
    }

    //--- Convenience functions ---

    /// Get value of given entry
    virtual double operator() (uint i, uint j) const
    { double value(0); get(&value, 1, &i, 1, &j); return value; }

    /// Get value of given entry
    virtual double getitem(std::pair<uint, uint> ij) const
    { double value(0); get(&value, 1, &ij.first, 1, &ij.second); return value; }

    /// Set given entry to value. apply("insert") should be called before using
    /// using the object.
    virtual void setitem(std::pair<uint, uint> ij, double value)
    { set(&value, 1, &ij.first, 1, &ij.second); }

    /// Insert one on the diagonal for all zero rows
    virtual void ident_zeros();

    typedef XMLMatrix XMLHandler;

  };

}
#endif
