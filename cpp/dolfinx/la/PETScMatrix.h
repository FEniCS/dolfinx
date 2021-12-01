// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "PETScOperator.h"
#include "utils.h"
#include <functional>
#include <petscmat.h>
#include <string>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::la
{
class SparsityPattern;

namespace petsc
{
/// Create a PETSc Mat. Caller is responsible for destroying the
/// returned object.
Mat create_matrix(MPI_Comm comm, const SparsityPattern& sp,
                  const std::string& type = std::string());

/// Create PETSc MatNullSpace. Caller is responsible for destruction
/// returned object.
/// @param [in] comm The MPI communicator
/// @param[in] basis The nullspace basis vectors
/// @return A PETSc nullspace object
MatNullSpace create_nullspace(MPI_Comm comm, const xtl::span<const Vec>& basis);
} // namespace petsc

/// It is a simple wrapper for a PETSc matrix pointer (Mat). Its main
/// purpose is to assist memory management of PETSc Mat objects.
///
/// For advanced usage, access the PETSc Mat pointer using the function
/// mat() and use the standard PETSc interface.

class PETScMatrix : public PETScOperator
{
public:
  /// Return a function with an interface for adding or inserting values
  /// into the matrix A (calls MatSetValuesLocal)
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_fn(Mat A, InsertMode mode);

  /// Return a function with an interface for adding or inserting values
  /// into the matrix A using blocked indices
  /// (calls MatSetValuesBlockedLocal)
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_block_fn(Mat A, InsertMode mode);

  /// Return a function with an interface for adding or inserting blocked
  /// values to the matrix A using non-blocked insertion (calls
  /// MatSetValuesLocal). Internally it expands the blocked indices into
  /// non-blocked arrays.
  /// @param[in] A The matrix to set values in
  /// @param[in] bs0 Block size for the matrix rows
  /// @param[in] bs1 Block size for the matrix columns
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_block_expand_fn(Mat A, int bs0, int bs1, InsertMode mode);

  /// Create holder for a PETSc Mat object from a sparsity pattern
  PETScMatrix(MPI_Comm comm, const SparsityPattern& sp,
              const std::string& type = std::string());

  /// Create holder of a PETSc Mat object/pointer. The Mat A object
  /// should already be created. If inc_ref_count is true, the reference
  /// counter of the Mat will be increased. The Mat reference count will
  /// always be decreased upon destruction of the the PETScMatrix.
  PETScMatrix(Mat A, bool inc_ref_count);

  // Copy constructor (deleted)
  PETScMatrix(const PETScMatrix& A) = delete;

  /// Move constructor (falls through to base class move constructor)
  PETScMatrix(PETScMatrix&& A) = default;

  /// Destructor
  ~PETScMatrix() = default;

  /// Assignment operator (deleted)
  PETScMatrix& operator=(const PETScMatrix& A) = delete;

  /// Move assignment operator
  PETScMatrix& operator=(PETScMatrix&& A) = default;

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

  /// Return norm of matrix
  double norm(Norm norm_type) const;

  //--- Special PETSc Functions ---

  /// Sets the prefix used by PETSc when searching the options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function MatSetFromOptions on the PETSc Mat object
  void set_from_options();
};
} // namespace dolfinx::la
