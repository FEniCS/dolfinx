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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>

#include <dolfin/common/types.h>
#include "GenericMatrix.h"

namespace dolfin
{

  class GenericLinearAlgebraFactory;
  class IndexMap;
  class TensorLayout;
  class TpetraVector;

  /// This class provides a simple matrix class based on Tpetra.  It
  /// is a wrapper for a Tpetra matrix pointer
  /// (Teuchos::RCP<matrix_type>) implementing the GenericMatrix
  /// interface.
  ///
  /// The interface is intentionally simple. For advanced usage,
  /// access the Tpetra::RCP<matrix_type> pointer using the function
  /// mat() and use the standard Tpetra interface.

  class TpetraMatrix : public GenericMatrix
  {

  public:

    // Tpetra typedefs with default values
    typedef Tpetra::CrsMatrix<double, int, dolfin::la_index> matrix_type;
    typedef Tpetra::CrsGraph<int, dolfin::la_index> graph_type;
    typedef Tpetra::Map<int, dolfin::la_index> map_type;

    /// Create empty matrix
    TpetraMatrix();

    /// Create a wrapper around a Teuchos::RCP<matrix_type> pointer
    explicit TpetraMatrix(Teuchos::RCP<matrix_type> A);

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
    std::pair<std::int64_t, std::int64_t> local_range(std::size_t dim) const;

    // Number of non-zero entries
    std::size_t nnz() const;

    /// Set all entries to zero and keep any sparse structure
    virtual void zero();

    /// Finalize assembly of tensor. The mode parameter is ignored.
    virtual void apply(std::string mode);

    /// Return MPI communicator
    MPI_Comm mpi_comm() const;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

    //--- Implementation of the GenericMatrix interface --

    /// Return copy of matrix
    virtual std::shared_ptr<GenericMatrix> copy() const;

    /// Initialize vector z to be compatible with the matrix-vector
    /// product y = Ax. In the parallel case, both size and layout are
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

    /// Get diagonal of a matrix
    virtual void get_diagonal(GenericVector& x) const;

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

    Teuchos::RCP<matrix_type> mat()
    { return _matA; }

    Teuchos::RCP<const matrix_type> mat() const
    { return _matA; }

    static void graphdump(const Teuchos::RCP<const graph_type> graph);

  private:

    // The matrix
    Teuchos::RCP<matrix_type> _matA;

    // Row and Column maps to allow local indexing of off-process
    // entries needed in add_local() and set_local()
    std::array<std::shared_ptr<const IndexMap>, 2> index_map;

  };

}

#endif

#endif
