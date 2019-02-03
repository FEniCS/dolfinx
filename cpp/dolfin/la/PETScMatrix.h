// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "PETScOperator.h"
#include "utils.h"
#include <array>
#include <petscmat.h>
#include <string>

namespace dolfin
{

namespace la
{
class SparsityPattern;
class VectorSpaceBasis;

/// It is a simple wrapper for a PETSc matrix pointer (Mat). Its main
/// purpose is to assist memory management of PETSc Mat objects.
///
/// For advanced usage, access the PETSc Mat pointer using the function
/// mat() and use the standard PETSc interface.

class PETScMatrix : public PETScOperator
{
public:
  PETScMatrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern);

  /// Create holder of a PETSc Mat object/pointer. The Mat A object
  /// should already be created. If inc_ref_count is true, the reference
  /// counter of the Mat will be increased. The Mat reference count will
  /// always be decreased upon destruction of the the PETScMatrix.
  explicit PETScMatrix(Mat A, bool inc_ref_count = true);

  // Copy constructor (deleted)
  PETScMatrix(const PETScMatrix& A) = delete;

  /// Move constructor (falls through to base class move constructor)
  PETScMatrix(PETScMatrix&& A) = default;

  /// Destructor
  ~PETScMatrix();

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

  /// Set block of values using global indices
  void set(const PetscScalar* block, std::size_t m, const PetscInt* rows,
           std::size_t n, const PetscInt* cols);

  /// Add block of values using local indices
  void add_local(const PetscScalar* block, std::size_t m, const PetscInt* rows,
                 std::size_t n, const PetscInt* cols);

  /// Return norm of matrix
  double norm(la::Norm norm_type) const;

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
};
} // namespace la
} // namespace dolfin
