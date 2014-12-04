// Copyright (C) 2014
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
// First added:  2014

#ifndef __TPETRA_MATRIX_H
#define __TPETRA_MATRIX_H

#ifdef HAS_TRILINOS

#include <map>
#include <string>
#include <memory>

#include "GenericMatrix.h"

#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>

// Tpetra typedefs with default values
typedef Tpetra::Vector<>::scalar_type scalar_type;
typedef Tpetra::Vector<>::local_ordinal_type local_ordinal_type;
typedef Tpetra::Vector<>::global_ordinal_type global_ordinal_type;
typedef Tpetra::Vector<>::node_type node_type;
typedef Tpetra::Map<> map_type;
typedef Tpetra::Vector<> vector_type;
typedef Tpetra::CrsMatrix<> matrix_type;

namespace dolfin
{

  class TpetraVector;
  class TensorLayout;

  /// This class provides a simple matrix class based on Tpetra.
  /// It is a wrapper for a Tpetra matrix pointer (Teuchos::RCP<matrix_type>)
  /// implementing the GenericMatrix interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the Tpetra::RCP<matrix_type> pointer using the function mat() and
  /// use the standard Tpetra interface.

  class TpetraMatrix : public GenericMatrix
  {
  public:

    /// Create empty matrix
    TpetraMatrix();

    /// Create a wrapper around a Tpetra Mat pointer
    explicit TpetraMatrix(Tpetra::RCP<matrix_type> A);

    /// Copy constructor
    TpetraMatrix(const TpetraMatrix& A);

    /// Destructor
    virtual ~TpetraMatrix();

    //--- Implementation of the GenericTensor interface ---

    /// Initialize zero tensor using tensor layout
    void init(const TensorLayout& tensor_layout);

    /// Return true if empty
    bool empty() const;

    /// Return size of given dimension
    std::size_t size(std::size_t dim) const;

    /// Return local ownership range
    std::pair<std::size_t, std::size_t> local_range(std::size_t dim) const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor. The following values are recognized
    /// for the mode parameter:
    ///
    ///   add    - corresponds to Tpetra MatAssemblyBegin+End(MAT_FINAL_ASSEMBLY)
    ///   insert - corresponds to Tpetra MatAssemblyBegin+End(MAT_FINAL_ASSEMBLY)
    ///   flush  - corresponds to Tpetra MatAssemblyBegin+End(MAT_FLUSH_ASSEMBLY)
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
    /// *Arguments*
    ///     dim (std::size_t)
    ///         The dimension (axis): dim = 0 --> z = y, dim = 1 --> z = x
    virtual void init_vector(GenericVector& z, std::size_t dim) const;

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

    /// Set diagonal of a matrix
    virtual void set_diagonal(const GenericVector& x);

    /// Multiply matrix by given number
    virtual const TpetraMatrix& operator*= (double a);

    /// Divide matrix by given number
    virtual const TpetraMatrix& operator/= (double a);

    /// Assignment operator
    virtual const GenericMatrix& operator= (const GenericMatrix& A);

    /// Test if matrix is symmetric
    virtual bool is_symmetric(double tol) const;

    //--- Special functions ---

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const;

    //--- Special TpetraFunctions ---

    /// Return norm of matrix
    double norm(std::string norm_type) const;

    /// Assignment operator
    const TpetraMatrix& operator= (const TpetraMatrix& A);

  private:

    // Tpetra norm types
    //    static const std::map<std::string, NormType> norm_types;

    Teuchos::RCP<matrix_type> _matA;

  };

}

#endif

#endif
