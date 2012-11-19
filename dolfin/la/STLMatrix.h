// Copyright (C) 2007-2012 Anders Logg
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
// Modified by Ola Skavhaug 2007
// Modified by Garth N. Wells 2007, 2009
// Modified by Ilmar Wilbers 2008
//
// First added:  2007-01-17
// Last changed: 2012-08-20

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
    STLMatrix(unsigned int primary_dim=0) : primary_dim(primary_dim),
      _local_range(0, 0), num_codim_entities(0) {}

    /// Destructor
    virtual ~STLMatrix() {}

    ///--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const TensorLayout& tensor_layout);

    /// Return size of given dimension
    virtual std::size_t size(unsigned int dim) const;

    /// Return local ownership range
    virtual std::pair<std::size_t, std::size_t> local_range(unsigned int dim) const;

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

    /// Resize vector z to be compatible with the matrix-vector product
    /// y = Ax. In the parallel case, both size and layout are
    /// important.
    ///
    /// *Arguments*
    ///     dim (unsigned int)
    ///         The dimension (axis): dim = 0 --> z = y, dim = 1 --> z = x
    virtual void resize(GenericVector& z, unsigned int dim) const
    { dolfin_not_implemented(); }

    /// Get block of values
    virtual void get(double* block, std::size_t m, const DolfinIndex* rows, std::size_t n,
                     const DolfinIndex* cols) const
    { dolfin_not_implemented(); }

    /// Set block of values
    virtual void set(const double* block, std::size_t m, const DolfinIndex* rows, std::size_t n,
                     const DolfinIndex* cols)
    { dolfin_not_implemented(); }

    /// Add block of values
    virtual void add(const double* block, std::size_t m, const DolfinIndex* rows, std::size_t n,
                     const DolfinIndex* cols);

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A,
                      bool same_nonzero_pattern)
    { dolfin_not_implemented(); }

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const;

    /// Get non-zero values of given row
    virtual void getrow(std::size_t row, std::vector<std::size_t>& columns,
                        std::vector<double>& values) const;

    /// Set values for given row
    virtual void setrow(std::size_t row, const std::vector<std::size_t>& columns,
                        const std::vector<double>& values)
    { dolfin_not_implemented(); }

    /// Set given rows to zero
    virtual void zero(std::size_t m, const DolfinIndex* rows)
    { dolfin_not_implemented(); }

    /// Set given rows to identity matrix
    virtual void ident(std::size_t m, const DolfinIndex* rows);

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
    virtual GenericLinearAlgebraFactory& factory() const;

    ///--- STLMatrix interface ---

    void sort()
    {
      std::vector<std::vector<std::pair<std::size_t, double> > >::iterator row;
      for (row = _values.begin(); row < _values.end(); ++row)
        std::sort(row->begin(), row->end());
    }

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
    std::size_t nnz() const;

    /// Return number of local non-zero entries
    std::size_t local_nnz() const;

  private:

    /// Return matrix in compressed format
    template<typename T>
    void compressed_storage(std::vector<double>& vals,
                            std::vector<T>& rows,
                            std::vector<T>& col_ptr,
                            std::vector<T>& local_to_global_col,
                            bool symmetric) const;

    // Primary dimension (0=row-wise storage, 1=column-wise storage)
    const unsigned int primary_dim;

    // Local ownership range (row range for row-wise storage, column
    // range for column-wise storage)
    std::pair<std::size_t, std::size_t> _local_range;

    // Number of columns (row-wise storage) or number of rows (column-wise
    // storage)
    std::size_t num_codim_entities;

    // Storage of non-zero matrix values
    std::vector<std::vector<std::pair<std::size_t, double> > > _values;

    // Off-process data ([i, j], value)
    boost::unordered_map<std::pair<std::size_t, std::size_t>, double> off_processs_data;

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
    row_ptr.reserve(_values.size() + 1);
    local_to_global_row.reserve(_values.size());

    // Build CSR data structures
    row_ptr.push_back(0);

    // Number of local non-zero entries
    const std::size_t _local_nnz = local_nnz();

    // Number of local rows (columns)
    const std::size_t num_local_rows = _values.size();

    if (!symmetric)
    {
      // Reserve memory
      vals.reserve(_local_nnz);
      cols.reserve(_local_nnz);

      // Build data structures
      for (std::size_t local_row = 0; local_row < num_local_rows; ++local_row)
      {
        for (std::size_t column = 0; column < _values[local_row].size(); ++column)
        {
          cols.push_back(_values[local_row][column].first);
          vals.push_back(_values[local_row][column].second);

          row_ptr.push_back(row_ptr.back() + _values[local_row].size());
          local_to_global_row.push_back(_local_range.first + local_row);
        }
      }
    }
    else
    {
      // Reserve memory
      vals.reserve((_local_nnz - num_local_rows)/2 + num_local_rows);
      cols.reserve((_local_nnz - num_local_rows)/2 + num_local_rows);

      // Build data structures
      for (std::size_t local_row = 0; local_row < _values.size(); ++local_row)
      {
        const std::size_t global_row_index = local_row + _local_range.first;
        std::size_t counter = 0;
        for (std::size_t i = 0; i < _values[local_row].size(); ++i)
        {
          const std::size_t index = _values[local_row][i].first;
          if (index >= global_row_index)
          {
            vals.push_back(_values[local_row][i].second);
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
