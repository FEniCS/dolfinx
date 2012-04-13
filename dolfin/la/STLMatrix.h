// Copyright (C) 2007-2010 Anders Logg
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
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007, 2009.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2011-10-29

#ifndef __DOLFIN_STL_MATRIX_H
#define __DOLFIN_STL_MATRIX_H

#include <string>
#include <utility>
#include <vector>
#include <boost/unordered_map.hpp>

#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include "TensorLayout.h"
#include "GenericMatrix.h"

namespace dolfin
{

  class GenericSparsityPattern;
  class GenericVector;

  /// Simple STL-based implementation of the GenericMatrix interface.
  /// The sparse matrix is stored as a pair of std::vector of
  /// std::vector, one for the columns and one for the values.
  ///
  /// Historically, this class has undergone a number of different
  /// incarnations, based on various combinations of std::vector,
  /// std::set and std::map. The current implementation has proven to
  /// be the fastest.

  class STLMatrix : public GenericMatrix
  {
  public:

    /// Create empty matrix
    STLMatrix(uint primary_dim=0) : primary_dim(primary_dim),
      _local_range(0, 0), num_codim_entities(0) {}

    /// Destructor
    virtual ~STLMatrix() {}

    ///--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const TensorLayout& tensor_layout);

    /// Return size of given dimension
    virtual uint size(uint dim) const;

    /// Return local ownership range
    virtual std::pair<uint, uint> local_range(uint dim) const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericMatrix interface ---

    /// Return copy of matrix
    virtual boost::shared_ptr<GenericMatrix> copy() const
    {
      boost::shared_ptr<GenericMatrix> A(new STLMatrix(*this));
      return A;
    }

    /// Resize vector y such that is it compatible with matrix for
    /// multuplication Ax = b (dim = 0 -> b, dim = 1 -> x) In parallel
    /// case, size and layout are important.
    virtual void resize(GenericVector& y, uint dim) const
    { dolfin_not_implemented(); }

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows, uint n,
                     const uint* cols) const
    { dolfin_not_implemented(); }

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows, uint n,
                     const uint* cols)
    { dolfin_not_implemented(); }

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows, uint n,
                     const uint* cols);

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A,
                      bool same_nonzero_pattern)
    { dolfin_not_implemented(); }

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const;

    /// Get non-zero values of given row
    virtual void getrow(uint row, std::vector<uint>& columns,
                        std::vector<double>& values) const;

    /// Set values for given row
    virtual void setrow(uint row, const std::vector<uint>& columns,
                        const std::vector<double>& values)
    { dolfin_not_implemented(); }

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows)
    { dolfin_not_implemented(); }

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows);

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const
    { dolfin_not_implemented(); }

    // Matrix-vector product, y = A^T x
    virtual void transpmult(const GenericVector& x, GenericVector& y) const
    { dolfin_not_implemented(); }

    /// Multiply matrix by given number
    virtual const STLMatrix& operator*= (double a);

    /// Divide matrix by given number
    virtual const STLMatrix& operator/= (double a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A)
    { dolfin_not_implemented(); return *this; }

    ///--- Specialized matrix functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    ///--- STLMatrix interface ---

    /// Return matrix in CSR format
    void csr(std::vector<double>& vals, std::vector<uint>& cols,
             std::vector<uint>& row_ptr,
             std::vector<uint>& local_to_global_row,
             bool base_one = false) const;

    /// Return matrix in CSC format
    void csc(std::vector<double>& vals, std::vector<uint>& rows,
             std::vector<uint>& col_ptr,
             std::vector<uint>& local_to_global_col,
             bool base_one = false) const;

    /// Return number of global non-zero entries
    uint nnz() const;

    /// Return number of local non-zero entries
    uint local_nnz() const;

  private:

    /// Return matrix in compressed format
    void compressed_storage(std::vector<double>& vals,
                            std::vector<uint>& rows,
                            std::vector<uint>& col_ptr,
                            std::vector<uint>& local_to_global_col,
                            bool base_one) const;

    // Primary dimension (0=row-wise storage, 1=column-wise storage)
    const uint primary_dim;

    // Local ownership range (row range for row-wise storage, column
    // range for column-wise storage)
    std::pair<uint, uint> _local_range;

    // Number of columns (row-wise storage) or number of rows (column-wise
    // storage)
    uint num_codim_entities;

    // Storage of columns (row-wise storgae) / row (column-wise storgae)
    // indices
    std::vector<std::vector<uint> > codim_indices;

    // Storage of values
    std::vector<std::vector<double> > _vals;

    // Off-process data ([i, j], value)
    boost::unordered_map<std::pair<uint, uint>, double> off_processs_data;

  };

}

#endif
