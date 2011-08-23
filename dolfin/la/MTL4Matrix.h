// Copyright (C) 2008 Dag Lindbo
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
// Modified by Garth N. Wells, 2008, 2009.
//
// First added:  2008-07-06
// Last changed: 2009-09-08

#ifdef HAS_MTL4

#ifndef __MTL4_MATRIX_H
#define __MTL4_MATRIX_H

#include <utility>
#include <tr1/tuple>
#include "GenericMatrix.h"
#include "mtl4.h"

//  Developers note:
//
//  This class implements a minimal backend for MTL4.
//
//  There are certain inline decisions that have been deferred.
//  Due to the extensive calling of this backend through the generic LA
//  interface, it is not clear where inlining will be possible and
//  improve performance.

namespace dolfin
{

  class MTL4Matrix: public GenericMatrix
  {
  public:

    /// Create empty matrix
    MTL4Matrix();

    /// Copy constuctor
    MTL4Matrix(const MTL4Matrix& A);

    /// Destructor
    virtual ~MTL4Matrix();

    //--- Implementation of the GenericTensor interface ---

    /// Return true if matrix is distributed
    virtual bool distributed() const
    { return false; }

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern);

    /// Return copy of tensor
    virtual MTL4Matrix* copy() const;

    /// Return size of given dimension
    virtual uint size(uint dim) const;

    /// Return local ownership range
    virtual std::pair<uint, uint> local_range(uint dim) const
    { return std::make_pair(0, size(dim)); }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericMatrix interface ---

    /// Resize matrix to M x N
    virtual void resize(uint M, uint N);

    /// Resize vector y such that is it compatible with matrix for
    /// multuplication Ax = b (dim = 0 -> b, dim = 1 -> x) In parallel
    /// case, size and layout are important.
    virtual void resize(GenericVector& y, uint dim) const;

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows, uint n, const uint* cols) const;

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A,  bool same_nonzero_pattern);

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const;

    /// Get non-zero values of given row
    virtual void getrow(uint row, std::vector<uint>& columns, std::vector<double>& values) const;

    /// Set values for given row
    virtual void setrow(uint row, const std::vector<uint>& columns, const std::vector<double>& values);

    /// Set given rows to zero
    virtual void zero(uint m, const uint* rows);

    /// Set given rows to identity matrix
    virtual void ident(uint m, const uint* rows);

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const;

    // Matrix-vector product, y = A^T x
    virtual void transpmult(const GenericVector& x, GenericVector& y) const;

    /// Multiply matrix by given number
    virtual const MTL4Matrix& operator*= (double a);

    /// Divide matrix by given number
    virtual const MTL4Matrix& operator/= (double a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A);

    /// Return pointers to underlying compresssed storage data
    /// See GenericMatrix for documentation.
    virtual std::tr1::tuple<const std::size_t*, const std::size_t*, const double*, int> data() const;

    //--- Special functions ---

    virtual LinearAlgebraFactory& factory() const;

    //--- Special MTL4 functions ---

    /// Create M x N matrix with estimate of nonzeroes per row
    MTL4Matrix(uint M, uint N, uint nz);

    /// Return mtl4_sparse_matrix reference
    const mtl4_sparse_matrix& mat() const;

    mtl4_sparse_matrix& mat();

    /// Assignment operator
    const MTL4Matrix& operator= (const MTL4Matrix& A);

  private:

    void init_inserter(uint nnz);
    void assert_no_inserter() const;

    // MTL4 matrix object
    mtl4_sparse_matrix A;

    // MTL4 matrix inserter
    mtl::matrix::inserter<mtl4_sparse_matrix, mtl::update_plus<double> >* ins;

    uint nnz_row;
  };

}

#endif
#endif
