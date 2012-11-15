// Copyright (C) 2006-2008 Anders Logg and Garth N. Wells
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
// Modified by Ola Skavhaug, 2007-2008.
// Modified by Kent-Andre Mardal, 2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2006-05-15
// Last changed: 2009-09-08

#ifndef __MATRIX_H
#define __MATRIX_H

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include "DefaultFactory.h"
#include "GenericMatrix.h"

namespace dolfin
{

  class TensorLayout;

  /// This class provides the default DOLFIN matrix class,
  /// based on the default DOLFIN linear algebra backend.

  class Matrix : public GenericMatrix
  {
  public:

    /// Create empty matrix
    Matrix()
    {
      DefaultFactory factory;
      matrix = factory.create_matrix();
    }

    /// Copy constructor
    Matrix(const Matrix& A) : matrix(A.matrix->copy()) {}

    /// Create a Vector from a GenericVetor
    Matrix(const GenericMatrix& A) : matrix(A.copy()) {}

    /// Destructor
    virtual ~Matrix() {}

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using tensor layout
    virtual void init(const TensorLayout& tensor_layout)
    { matrix->init(tensor_layout); }

    /// Return size of given dimension
    virtual std::size_t size(uint dim) const
    { return matrix->size(dim); }

    /// Return local ownership range
    virtual std::pair<std::size_t, std::size_t> local_range(uint dim) const
    { return matrix->local_range(dim); }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    { matrix->zero(); }

    /// Finalize assembly of tensor
    virtual void apply(std::string mode)
    { matrix->apply(mode); }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const
    { return "<Matrix wrapper of " + matrix->str(verbose) + ">"; }

    //--- Implementation of the GenericMatrix interface ---

    /// Return copy of matrix
    virtual boost::shared_ptr<GenericMatrix> copy() const
    {
      boost::shared_ptr<Matrix> A(new Matrix(*this));
      return A;
    }

    /// Resize vector y such that is it compatible with matrix for
    /// multuplication Ax = b (dim = 0 -> b, dim = 1 -> x) In parallel
    /// case, size and layout are important.
    virtual void resize(GenericVector& y, uint dim) const
    { matrix->resize(y, dim); }

    /// Get block of values
    virtual void get(double* block, std::size_t m, const DolfinIndex* rows, std::size_t n, const DolfinIndex* cols) const
    { matrix->get(block, m, rows, n, cols); }

    /// Set block of values
    virtual void set(const double* block, std::size_t m, const DolfinIndex* rows, std::size_t n, const DolfinIndex* cols)
    { matrix->set(block, m, rows, n, cols); }

    /// Add block of values
    virtual void add(const double* block, std::size_t m, const DolfinIndex* rows, std::size_t n, const DolfinIndex* cols)
    { matrix->add(block, m, rows, n, cols); }

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A, bool same_nonzero_pattern)
    { matrix->axpy(a, A, same_nonzero_pattern); }

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const
    { return matrix->norm(norm_type); }

    /// Get non-zero values of given row
    virtual void getrow(std::size_t row, std::vector<DolfinIndex>& columns, std::vector<double>& values) const
    { matrix->getrow(row, columns, values); }

    /// Set values for given row
    virtual void setrow(std::size_t row, const std::vector<DolfinIndex>& columns, const std::vector<double>& values)
    { matrix->setrow(row, columns, values); }

    /// Set given rows to zero
    virtual void zero(std::size_t m, const DolfinIndex* rows)
    { matrix->zero(m, rows); }

    /// Set given rows to identity matrix
    virtual void ident(std::size_t m, const DolfinIndex* rows)
    { matrix->ident(m, rows); }

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const
    { matrix->mult(x, y); }

    // Matrix-vector product, y = Ax
    virtual void transpmult(const GenericVector& x, GenericVector& y) const
    { matrix->transpmult(x, y); }

    /// Multiply matrix by given number
    virtual const Matrix& operator*= (double a)
    { *matrix *= a; return *this; }

    /// Divide matrix by given number
    virtual const Matrix& operator/= (double a)
    { *matrix /= a; return *this; }

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A)
    { *matrix = A; return *this; }

    /// Return pointers to underlying compressed storage data.
    /// See GenericMatrix for documentation.
    virtual boost::tuples::tuple<const std::size_t*, const std::size_t*, const double*, int> data() const
    { return matrix->data(); }

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const
    { return matrix->factory(); }

    //--- Special functions, intended for library use only ---

    /// Return concrete instance / unwrap (const version)
    virtual const GenericMatrix* instance() const
    { return matrix.get() ; }

    /// Return concrete instance / unwrap (non-const version)
    virtual GenericMatrix* instance()
    { return matrix.get(); }

    virtual boost::shared_ptr<const LinearAlgebraObject> shared_instance() const
    { return matrix; }

    virtual boost::shared_ptr<LinearAlgebraObject> shared_instance()
    { return matrix; }

    //--- Special Matrix functions ---

    /// Assignment operator
    const Matrix& operator= (const Matrix& A)
    { *matrix = *A.matrix; return *this; }

  private:

    // Pointer to concrete implementation
    boost::shared_ptr<GenericMatrix> matrix;

  };

}

#endif
