// Copyright (C) 2004-2012 Johan Hoffman, Johan Jansson, Anders Logg
// and Garth N. Wells
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
// Modified by Andy R. Terrel 2005
// Modified by Garth N. Wells 2006-2009
// Modified by Kent-Andre Mardal 2008
// Modified by Ola Skavhaug 2008
// Modified by Fredrik Valdmanis 2011
//
// First added:  2004-01-01
// Last changed: 2012-08-22

#ifndef __PETSC_MATRIX_H
#define __PETSC_MATRIX_H

#ifdef HAS_PETSC

#include <map>
#include <memory>
#include <string>

#include <petscmat.h>
#include <petscsys.h>

#include "GenericMatrix.h"
#include "PETScBaseMatrix.h"

namespace dolfin
{

  class PETScVector;
  class TensorLayout;
  class VectorSpaceBasis;

  /// This class provides a simple matrix class based on PETSc.
  /// It is a wrapper for a PETSc matrix pointer (Mat)
  /// implementing the GenericMatrix interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the PETSc Mat pointer using the function mat() and
  /// use the standard PETSc interface.

  class PETScMatrix : public GenericMatrix, public PETScBaseMatrix
  {
  public:

    /// Create empty matrix (on MPI_COMM_WORLD)
    PETScMatrix();

    /// Create empty matrix
    explicit PETScMatrix(MPI_Comm comm);

    /// Create a wrapper around a PETSc Mat pointer. The Mat object
    /// should have been created, e.g. via PETSc MatrCreate.
    explicit PETScMatrix(Mat A);

    /// Copy constructor
    PETScMatrix(const PETScMatrix& A);

    /// Destructor
    virtual ~PETScMatrix();

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using tensor layout
    void init(const TensorLayout& tensor_layout);

    /// Return true if empty
    bool empty() const;

    /// Return size of given dimension
    std::size_t size(std::size_t dim) const
    { return PETScBaseMatrix::size(dim); }

    /// Return local ownership range
    std::pair<std::int64_t, std::int64_t> local_range(std::size_t dim) const
    { return PETScBaseMatrix::local_range(dim); }

    /// Return number of non-zero entries in matrix (collective)
    std::size_t nnz() const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor. The following values are recognized
    /// for the mode parameter:
    ///
    ///   add    - corresponds to PETSc MatAssemblyBegin+End(MAT_FINAL_ASSEMBLY)
    ///   insert - corresponds to PETSc MatAssemblyBegin+End(MAT_FINAL_ASSEMBLY)
    ///   flush  - corresponds to PETSc MatAssemblyBegin+End(MAT_FLUSH_ASSEMBLY)
    virtual void apply(std::string mode);

    /// Return MPI communicator
    MPI_Comm mpi_comm() const;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericMatrix interface --

    /// Return copy of matrix
    virtual std::shared_ptr<GenericMatrix> copy() const;

    /// Initialize vector z to be compatible with the matrix-vector product
    /// y = Ax. In the parallel case, both size and layout are
    /// important.
    ///
    /// @param z (GenericVector&)
    ///         Vector to initialise
    /// @param  dim (std::size_t)
    ///         The dimension (axis): dim = 0 --> z = y, dim = 1 --> z = x
    virtual void init_vector(GenericVector& z, std::size_t dim) const
    { PETScBaseMatrix::init_vector(z, dim); }

    /// Get block of values
    virtual void get(double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols) const;

    /// Set block of values using global indices
    virtual void set(const double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols);

    /// Set block of values using local indices
    virtual void set_local(const double* block,
                           std::size_t m, const dolfin::la_index* rows,
                           std::size_t n, const dolfin::la_index* cols);

    /// Add block of values using global indices
    virtual void add(const double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols);

    /// Add block of values using local indices
    virtual void add_local(const double* block,
                           std::size_t m, const dolfin::la_index* rows,
                           std::size_t n, const dolfin::la_index* cols);

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A,
                      bool same_nonzero_pattern);

    /// Return norm of matrix
    double norm(std::string norm_type) const;

    /// Get non-zero values of given row
    virtual void getrow(std::size_t row,
                        std::vector<std::size_t>& columns,
                        std::vector<double>& values) const;

    /// Set values for given row
    virtual void setrow(std::size_t row,
                        const std::vector<std::size_t>& columns,
                        const std::vector<double>& values);

    /// Set given rows (global row indices) to zero
    virtual void zero(std::size_t m, const dolfin::la_index* rows);

    /// Set given rows (local row indices) to zero
    virtual void zero_local(std::size_t m, const dolfin::la_index* rows);

    /// Set given rows (global row indices) to identity matrix
    virtual void ident(std::size_t m, const dolfin::la_index* rows);

    /// Set given rows (local row indices) to identity matrix
    virtual void ident_local(std::size_t m, const dolfin::la_index* rows);

    // Matrix-vector product, y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const;

    // Matrix-vector product, y = A^T x
    virtual void transpmult(const GenericVector& x, GenericVector& y) const;

    /// Get diagonal of a matrix
    virtual void get_diagonal(GenericVector& x) const;

    /// Set diagonal of a matrix
    virtual void set_diagonal(const GenericVector& x);

    /// Multiply matrix by given number
    virtual const PETScMatrix& operator*= (double a);

    /// Divide matrix by given number
    virtual const PETScMatrix& operator/= (double a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A);

    /// Test if matrix is symmetric
    virtual bool is_symmetric(double tol) const;

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const;

    //--- Special PETSc Functions ---

    /// Sets the prefix used by PETSc when searching the options
    /// database
    void set_options_prefix(std::string options_prefix);

    /// Returns the prefix used by PETSc when searching the options
    /// database
    std::string get_options_prefix() const;

    /// Call PETSc function MatSetFromOptions on the PETSc Mat object
    void set_from_options();

    /// Assignment operator
    const PETScMatrix& operator= (const PETScMatrix& A);

    /// Attach nullspace to matrix (typically used by Krylov solvers
    /// when solving singular systems)
    void set_nullspace(const VectorSpaceBasis& nullspace);

    /// Attach near nullspace to matrix (used by preconditioners, such
    /// as smoothed aggregation algerbraic multigrid)
    void set_near_nullspace(const VectorSpaceBasis& nullspace);

    /// Dump matrix to PETSc binary format
    void binary_dump(std::string file_name) const;

  private:

    // Create PETSc nullspace object
    MatNullSpace create_petsc_nullspace(const VectorSpaceBasis& nullspace) const;

    // PETSc norm types
    static const std::map<std::string, NormType> norm_types;

  };

}

#endif

#endif
