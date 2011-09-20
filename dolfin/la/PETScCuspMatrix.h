// Copyright (C) 2011 Fredrik Valdmanis 
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
// First added:  2011-09-13
// Last changed: 2011-09-13

#ifndef __PETSC_CUSP_MATRIX_H
#define __PETSC_CUSP_MATRIX_H

//#ifdef PETSC_HAVE_CUSP // FIXME: Find a functioning test

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>
#include <petscmat.h>
#include "GenericMatrix.h"
#include "PETScBaseMatrix.h"

namespace dolfin
{

  class PETScCuspVector;

  /// This class provides a simple matrix class based on PETSc.
  /// It is a wrapper for a PETSc matrix pointer (Mat)
  /// implementing the GenericMatrix interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Mat pointer using the function mat() and
  /// use the standard PETSc interface.

  class PETScCuspMatrix : public GenericMatrix, public PETScBaseMatrix
  {
  public:

    /// Create empty matrix
    PETScCuspMatrix();

    /// Copy constructor
    PETScCuspMatrix(const PETScCuspMatrix& A);

    /// Create matrix from given PETSc Mat pointer
    explicit PETScCuspMatrix(boost::shared_ptr<Mat> A);

    /// Destructor
    virtual ~PETScCuspMatrix();

    //--- Implementation of the GenericTensor interface ---

    /// Return true if matrix is distributed
    bool distributed() const;

    /// Initialize zero tensor using sparsity pattern
    virtual void init(const GenericSparsityPattern& sparsity_pattern);

    /// Return copy of tensor
    virtual PETScCuspMatrix* copy() const;

    /// Return size of given dimension
    uint size(uint dim) const { return PETScBaseMatrix::size(dim); }

    /// Return local ownership range
    std::pair<uint, uint> local_range(uint dim) const
    { return PETScBaseMatrix::local_range(dim); };

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor
    virtual void apply(std::string mode);

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericMatrix interface --

    /// Resize matrix to M x N
    virtual void resize(uint M, uint N);

    /// Resize vector y such that is it compatible with matrix for
    /// multuplication Ax = b (dim = 0 -> b, dim = 1 -> x) In parallel
    /// case, size and layout are important.
    void resize(GenericVector& y, uint dim) const
    { PETScBaseMatrix::resize(y, dim); }

    /// Get block of values
    virtual void get(double* block, uint m, const uint* rows, uint n, const uint* cols) const;

    /// Set block of values
    virtual void set(const double* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add block of values
    virtual void add(const double* block, uint m, const uint* rows, uint n, const uint* cols);

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A, bool same_nonzero_pattern);

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
    virtual const PETScCuspMatrix& operator*= (double a);

    /// Divide matrix by given number
    virtual const PETScCuspMatrix& operator/= (double a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A);

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual LinearAlgebraFactory& factory() const;

    //--- Special PETScFunctions ---

    /// Return norm of matrix
    double norm(std::string norm_type) const;

    /// Assignment operator
    const PETScCuspMatrix& operator= (const PETScCuspMatrix& A);

    /// Dump matrix to PETSc binary format
    void binary_dump(std::string file_name) const;

  private:

    // PETSc norm types
    static const std::map<std::string, NormType> norm_types;

  };

}

#endif

//#endif
