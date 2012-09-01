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
    template<typename T>
    void csr(std::vector<double>& vals, std::vector<T>& cols,
             std::vector<T>& row_ptr,
             std::vector<T>& local_to_global_row,
             bool symmetric) const;

    /// Return matrix in CSC format
    template<typename T>
    void csc(std::vector<double>& vals, std::vector<T>& rows,
             std::vector<T>& col_ptr,
             std::vector<T>& local_to_global_col,
             bool symmetric) const;

    /// Return number of global non-zero entries
    uint nnz() const;

    /// Return number of local non-zero entries
    uint local_nnz() const;

  private:

    /// Return matrix in compressed format
    template<typename T>
    void compressed_storage(std::vector<double>& vals,
                            std::vector<T>& rows,
                            std::vector<T>& col_ptr,
                            std::vector<T>& local_to_global_col,
                            bool symmetric) const;

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

  //---------------------------------------------------------------------------
  // Implementation of templated functions
  //---------------------------------------------------------------------------
  template<typename T>
  void STLMatrix::csr(std::vector<double>& vals, std::vector<T>& cols,
                      std::vector<T>& row_ptr,
                      std::vector<T>& local_to_global_row,
                      bool symmetric) const
  {
    if (primary_dim != 0)
    {
      dolfin_error("STLMatrix.cpp",
                   "creating compressed row storage data",
                   "Cannot create CSR matrix from STLMatrix with column-wise storage.");
    }
    compressed_storage(vals, cols, row_ptr, local_to_global_row, symmetric);
  }
  //-----------------------------------------------------------------------------
  template<typename T>
  void STLMatrix::csc(std::vector<double>& vals, std::vector<T>& rows,
                      std::vector<T>& col_ptr,
                      std::vector<T>& local_to_global_col,
                      bool symmetric) const
  {
    if (primary_dim != 1)
    {
      dolfin_error("STLMatrix.cpp",
                   "creating compressed column storage data",
                   "Cannot create CSC matrix from STLMatrix with row-wise storage.");
    }
    compressed_storage(vals, rows, col_ptr, local_to_global_col, symmetric);
  }
  //-----------------------------------------------------------------------------
  template<typename T>
  void STLMatrix::compressed_storage(std::vector<double>& vals,
                                     std::vector<T>& cols,
                                     std::vector<T>& row_ptr,
                                     std::vector<T>& local_to_global_row,
                                     bool symmetric) const
  {
    // Reset data structures
    vals.clear();
    cols.clear();
    row_ptr.clear();
    local_to_global_row.clear();

    // Reserve memory
    row_ptr.reserve(codim_indices.size() + 1);
    local_to_global_row.reserve(codim_indices.size());

    // Build CSR data structures
    row_ptr.push_back(0);

    // Number of local non-zero entries
    const std::size_t _local_nnz = local_nnz();

    // Number of local rows (columns)
    const std::size_t num_local_rows = codim_indices.size();

    if (!symmetric)
    {
      // Reserve memory
      vals.reserve(_local_nnz);
      cols.reserve(_local_nnz);

      // Build data structures
      for (std::size_t local_row = 0; local_row < num_local_rows; ++local_row)
      {
        vals.insert(vals.end(), _vals[local_row].begin(), _vals[local_row].end());
        cols.insert(cols.end(), codim_indices[local_row].begin(), codim_indices[local_row].end());

        row_ptr.push_back(row_ptr.back() + codim_indices[local_row].size());
        local_to_global_row.push_back(_local_range.first + local_row);
      }
    }
    else
    {
      // Reserve memory
      vals.reserve((_local_nnz - num_local_rows)/2 + num_local_rows);
      cols.reserve((_local_nnz - num_local_rows)/2 + num_local_rows);

      // Build data structures
      for (std::size_t local_row = 0; local_row < codim_indices.size(); ++local_row)
      {
        const uint global_row_index = local_row + _local_range.first;
        std::size_t counter = 0;
        for (std::size_t i = 0; i < codim_indices[local_row].size(); ++i)
        {
          const std::size_t index = codim_indices[local_row][i];
          if (index >= global_row_index)
          {
            vals.push_back(_vals[local_row][i]);
            cols.push_back(index);
            ++counter;
          }
        }

        row_ptr.push_back(row_ptr.back() + counter);
        local_to_global_row.push_back(global_row_index);
      }
    }
  }
//-----------------------------------------------------------------------------

}

#endif
