// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "PETScOperator.h"
#include "utils.h"
#include <dolfin/common/types.h>
#include <map>
#include <memory>
#include <petscmat.h>
#include <petscsys.h>
#include <string>

namespace dolfin
{

class PETScVector;

namespace la
{
class SparsityPattern;
class VectorSpaceBasis;

/// This class provides a simple matrix class based on PETSc.
/// It is a wrapper for a PETSc matrix pointer (Mat)
/// implementing the GenericMatrix interface.
///
/// The interface is intentionally simple. For advanced usage,
/// access the PETSc Mat pointer using the function mat() and
/// use the standard PETSc interface.

class PETScMatrix : public PETScOperator
{
public:
  /// Create empty matrix
  explicit PETScMatrix(MPI_Comm comm);

  /// Create a wrapper around a PETSc Mat pointer. The Mat object
  /// should have been created, e.g. via PETSc MatCreate.
  explicit PETScMatrix(Mat A);

  /// Copy constructor
  PETScMatrix(const PETScMatrix& A);

  /// Move constructor (falls through to base class move constructor)
  PETScMatrix(PETScMatrix&& A) = default;

  /// Destructor
  ~PETScMatrix();

  /// Assignment operator (deleted)
  PETScMatrix& operator=(const PETScMatrix& A) = delete;

  /// Move assignment operator
  PETScMatrix& operator=(PETScMatrix&& A) = default;

  /// Initialize zero tensor using sparsity pattern
  void init(const SparsityPattern& sparsity_pattern);

  /// Return true if empty
  bool empty() const;

  /// Return local ownership range
  std::array<std::int64_t, 2> local_range(std::size_t dim) const;

  /// Return number of non-zero entries in matrix (collective)
  std::size_t nnz() const;

  /// Set all entries to zero and keep any sparse structure
  void zero();

  /// Assembly type
  ///   FINAL - corresponds to PETSc MAT_FINAL_ASSEMBLY
  ///   FLUSH - corresponds to PETSc MAT_FLUSH_ASSEMBLY
  enum class AssemblyType : std::int32_t
  {
    FINAL,
    FLUSH
  };

  /// Finalize assembly of tensor. The following values are recognized
  /// for the mode parameter:
  /// @param type
  ///   FINAL    - corresponds to PETSc MatAssemblyBegin+End(MAT_FINAL_ASSEMBLY)
  ///   FLUSH  - corresponds to PETSc MatAssemblyBegin+End(MAT_FLUSH_ASSEMBLY)
  void apply(AssemblyType type);

  /// Return MPI communicator
  MPI_Comm mpi_comm() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Set block of values using local indices
  void set_local(const PetscScalar* block, const dolfin::la_index_t* num_rows,
                 const dolfin::la_index_t* const* rows)
  {
    set_local(block, num_rows[0], rows[0], num_rows[1], rows[1]);
  }

  /// Add block of values using local indices
  void add_local(const PetscScalar* block, const dolfin::la_index_t* num_rows,
                 const dolfin::la_index_t* const* rows)
  {
    add_local(block, num_rows[0], rows[0], num_rows[1], rows[1]);
  }

  /// Get block of values
  void get(PetscScalar* block, std::size_t m, const dolfin::la_index_t* rows,
           std::size_t n, const dolfin::la_index_t* cols) const;

  /// Set block of values using global indices
  void set(const PetscScalar* block, std::size_t m,
           const dolfin::la_index_t* rows, std::size_t n,
           const dolfin::la_index_t* cols);

  /// Set block of values using local indices
  void set_local(const PetscScalar* block, std::size_t m,
                 const dolfin::la_index_t* rows, std::size_t n,
                 const dolfin::la_index_t* cols);

  /// Add block of values using global indices
  void add(const PetscScalar* block, std::size_t m,
           const dolfin::la_index_t* rows, std::size_t n,
           const dolfin::la_index_t* cols);

  /// Add block of values using local indices
  void add_local(const PetscScalar* block, std::size_t m,
                 const dolfin::la_index_t* rows, std::size_t n,
                 const dolfin::la_index_t* cols);

  /// Add multiple of given matrix (AXPY operation)
  void axpy(PetscScalar a, const PETScMatrix& A, bool same_nonzero_pattern);

  /// Return norm of matrix
  double norm(la::Norm norm_type) const;

  /// Set given rows (global row indices) to zero
  void zero(std::size_t m, const dolfin::la_index_t* rows);

  /// Zero given rows (local row indices), and set diagonal
  void zero_local(std::size_t m, const dolfin::la_index_t* rows,
                  PetscScalar diag);

  /// Matrix-vector product, y = Ax
  void mult(const PETScVector& x, PETScVector& y) const;

  /// Get diagonal of a matrix
  void get_diagonal(PETScVector& x) const;

  /// Set diagonal of a matrix
  void set_diagonal(const PETScVector& x);

  /// Multiply matrix by scalar
  void scale(PetscScalar a);

  /// Test if matrix is symmetric
  bool is_symmetric(double tol) const;

  /// Test if matrix is hermitian
  bool is_hermitian(double tol) const;

  //--- Special PETSc Functions ---

  /// Sets the prefix used by PETSc when searching the options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function MatSetFromOptions on the PETSc Mat object
  void set_from_options();

  /// Attach nullspace to matrix (typically used by Krylov solvers
  /// when solving singular systems)
  void set_nullspace(const la::VectorSpaceBasis& nullspace);

  /// Attach 'near' nullspace to matrix (used by preconditioners,
  /// such as smoothed aggregation algerbraic multigrid)
  void set_near_nullspace(const la::VectorSpaceBasis& nullspace);

private:
  // Create PETSc nullspace object
  MatNullSpace
  create_petsc_nullspace(const la::VectorSpaceBasis& nullspace) const;

  // PETSc norm types
  static const std::map<std::string, NormType> norm_types;
};
} // namespace la
} // namespace dolfin
