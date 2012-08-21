// Copyright (C) 2006-2011 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Johan Jansson 2006
// Modified by Anders Logg 2006-2012
// Modified by Ola Skavhaug 2007-2008
// Modified by Kent-Andre Mardal 2008
// Modified by Martin Alnæs 2008
// Modified by Mikael Mortensen 2011
//
// First added:  2006-04-24
// Last changed: 2012-08-20

#ifndef __GENERIC_MATRIX_H
#define __GENERIC_MATRIX_H

#include <boost/tuple/tuple.hpp>
#include <vector>
#include "GenericTensor.h"
#include "GenericLinearOperator.h"

namespace dolfin
{

  class GenericVector;
  class TensorLayout;

  /// This class defines a common interface for matrices.

  class GenericMatrix : public GenericTensor, public GenericLinearOperator
  {
  public:

    /// Destructor
    virtual ~GenericMatrix() {}

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using tensor layout
    virtual void init(const TensorLayout& tensor_layout) = 0;

    /// Return tensor rank (number of dimensions)
    virtual uint rank() const
    { return 2; }

    /// Return size of given dimension
    virtual uint size(uint dim) const = 0;

    /// Return local ownership range
    virtual std::pair<uint, uint> local_range(uint dim) const = 0;

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

    /// Add block of values
    virtual void add(const double* block, const std::vector<const std::vector<uint>* >& rows)
    { add(block, (rows[0])->size(), &(*rows[0])[0], (rows[1])->size(), &(*rows[1])[0]); }

    /// Add block of values
    virtual void add(const double* block, const std::vector<std::vector<uint> >& rows)
    { add(block, rows[0].size(), &(rows[0])[0], rows[1].size(), &(rows[1])[0]); }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero() = 0;

    /// Finalize assembly of tensor
    virtual void apply(std::string mode) = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

    //--- Matrix interface ---

    /// Return copy of matrix
    virtual boost::shared_ptr<GenericMatrix> copy() const = 0;

    /// Resize vector y such that is it compatible with the linear
    /// system Ax = b. In the parallel case, both size and layout are
    /// important.
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         The dimension (axis): dim = 0 --> b, dim = 1 --> x
    virtual void resize(GenericVector& y, uint dim) const = 0;

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

    /// Matrix-vector product, y = A^T x. The y vector must either be
    /// zero-sized or have correct size and parallel layout.
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
    virtual boost::tuples::tuple<const std::size_t*, const std::size_t*,
                            const double*, int> data() const
    {
      dolfin_error("GenericMatrix.h",
                   "return pointers to underlying matrix data",
                   "Not implemented by current linear algebra backend");
      return boost::tuples::tuple<const std::size_t*, const std::size_t*,
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

    /// Compress matrix
    virtual void compress();

  };

}

#endif
