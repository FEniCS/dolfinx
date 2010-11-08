// Copyright (C) 2007-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// Modified by Garth N. Wells, 2007, 2009.
// Modified by Ilmar Wilbers, 2008.
//
// First added:  2007-01-17
// Last changed: 2010-11-08

#ifndef __STL_MATRIX_H
#define __STL_MATRIX_H

#include <string>
#include <vector>
#include <dolfin/log/log.h>
#include "GenericSparsityPattern.h"
#include "GenericMatrix.h"

namespace dolfin
{

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
    STLMatrix()
    { dims[0] = dims[1] = 0; }

    /// Create M x N matrix
    STLMatrix(uint M, uint N)
    { resize(M, N); }

    /// Copy constructor
    STLMatrix(const STLMatrix& A)
    { dolfin_not_implemented(); }

    /// Destructor
    virtual ~STLMatrix() {}

    ///--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern)
    { resize(sparsity_pattern.size(0), sparsity_pattern.size(1)); }

    /// Return copy of tensor
    virtual STLMatrix* copy() const
    { dolfin_not_implemented(); return 0; }

    /// Return size of given dimension
    virtual uint size(uint dim) const
    { assert(dim < 2); return dims[dim]; }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    {
      for (std::vector<std::vector<double> >::iterator row = vals.begin(); row != vals.end(); ++row)
        std::fill(row->begin(), row->end(), 0);
    }

    /// Finalize assembly of tensor
    virtual void apply(std::string mode) {}

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericMatrix interface ---

    /// Initialize M x N matrix
    virtual void resize(uint M, uint N)
    {
      cols.clear();
      vals.clear();

      cols.resize(M);
      vals.resize(N);
    }

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows, uint n, const uint* cols) const
    { dolfin_not_implemented(); }

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows, uint n, const uint* cols)
    { dolfin_not_implemented(); }

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows, uint n, const uint* cols)
    {
      // Perform a simple linear search along each column. Otherwise,
      // append the value (calling push_back).

      // Iterate over rows
      uint pos = 0;
      for (uint i = 0; i < m; i++)
      {
        const uint I = rows[i];
        std::vector<uint>& rcols = this->cols[I];
        std::vector<double>& rvals = this->vals[I];

        // Iterate over columns
        for (uint j = 0; j < n; j++)
        {
          const uint J = cols[j];

          // Try to find column
          bool found = false;
          for (uint k = 0; k < rcols.size(); k++)
          {
            if (rcols[k] == J)
            {
              rvals[k] += block[pos++];
              found = true;
              break;
            }
          }

          // Append if not found
          if (!found)
          {
            rcols.push_back(J);
            rvals.push_back(block[pos++]);
          }
        }
      }
    }

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A, bool same_nonzero_pattern)
    { dolfin_not_implemented(); }

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const
    { dolfin_not_implemented(); return 0.0; }

    /// Get non-zero values of given row
    virtual void getrow(uint row, std::vector<uint>& columns, std::vector<double>& values) const
    {
      assert(row < dims[0]);
      columns = this->cols[row];
      values = this->vals[row];
    }

    /// Set values for given row
    virtual void setrow(uint row, const std::vector<uint>& columns, const std::vector<double>& values)
    { dolfin_not_implemented(); }

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows)
    { dolfin_not_implemented(); }

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows)
    { dolfin_not_implemented(); }

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const
    { dolfin_not_implemented(); }

    // Matrix-vector product, y = A^T x
    virtual void transpmult(const GenericVector& x, GenericVector& y) const
    { dolfin_not_implemented(); }

    /// Multiply matrix by given number
    virtual const STLMatrix& operator*= (double a)
    { dolfin_not_implemented(); return *this; }

    /// Divide matrix by given number
    virtual const STLMatrix& operator/= (double a)
    { dolfin_not_implemented(); return *this; }

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A)
    { dolfin_not_implemented(); return *this; }

    ///--- Specialized matrix functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    /// Resize tensor of given rank and dimensions
    virtual void resize(uint rank, const uint* dims, bool reset)
    {
      // Check that the rank is 2
      if (rank != 2)
        error("Illegal tensor rank (%d) for matrix. Rank must be 2.", rank);

      // Initialize matrix
      resize(dims[0], dims[1]);

      // Save dimensions
      this->dims[0] = dims[0];
      this->dims[1] = dims[1];
    }

  private:

    // Storages of columns
    std::vector<std::vector<uint> > cols;

    // Storage of values
    std::vector<std::vector<double> > vals;

    // The size of the matrix
    uint dims[2];

  };

}

#endif
