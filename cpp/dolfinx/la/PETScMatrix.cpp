// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScMatrix.h"
#include "VectorSpaceBasis.h"
#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <iostream>
#include <sstream>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                  const std::int32_t*, const PetscScalar*)>
PETScMatrix::add_fn(Mat A)
{
  return [A, cache = std::vector<PetscInt>()](
             std::int32_t m, const std::int32_t* rows, std::int32_t n,
             const std::int32_t* cols, const PetscScalar* vals) mutable {
    PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
    cache.resize(m + n);
    std::copy(rows, rows + m, cache.begin());
    std::copy(cols, cols + n, cache.begin() + m);
    const PetscInt *_rows = cache.data(), *_cols1 = _rows + m;
    ierr = MatSetValuesLocal(A, m, _rows, n, _cols, vals, ADD_VALUES);
#else
    ierr = MatSetValuesLocal(A, m, rows, n, cols, vals, ADD_VALUES);
#endif

#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
    return 0;
  };
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern)
    : PETScOperator(create_petsc_matrix(comm, sparsity_pattern), false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(Mat A, bool inc_ref_count)
    : PETScOperator(A, inc_ref_count)
{
  // Reference count to A is incremented in base class
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const PetscScalar* block, int m, const PetscInt* rows,
                      int n, const PetscInt* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValues(_matA, m, rows, n, cols, block, INSERT_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add_local(const PetscScalar* block, int m,
                            const PetscInt* rows, int n, const PetscInt* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValuesLocal(_matA, m, rows, n, cols, block, ADD_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValuesLocal");
}
//-----------------------------------------------------------------------------
double PETScMatrix::norm(la::Norm norm_type) const
{
  assert(_matA);
  PetscErrorCode ierr;
  double value = 0.0;
  switch (norm_type)
  {
  case la::Norm::l1:
    ierr = MatNorm(_matA, NORM_1, &value);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatNorm");
    break;
  case la::Norm::linf:
    ierr = MatNorm(_matA, NORM_INFINITY, &value);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatNorm");
    break;
  case la::Norm::frobenius:
    ierr = MatNorm(_matA, NORM_FROBENIUS, &value);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatNorm");
    break;
  default:
    throw std::runtime_error("Unknown PETSc Mat norm type");
  }

  return value;
}
//-----------------------------------------------------------------------------
void PETScMatrix::apply(AssemblyType type)
{
  common::Timer timer("Apply (PETScMatrix)");

  assert(_matA);
  PetscErrorCode ierr;

  MatAssemblyType petsc_type = MAT_FINAL_ASSEMBLY;
  if (type == AssemblyType::FLUSH)
    petsc_type = MAT_FLUSH_ASSEMBLY;

  ierr = MatAssemblyBegin(_matA, petsc_type);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(_matA, petsc_type);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatAssemblyEnd");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_options_prefix(std::string options_prefix)
{
  assert(_matA);
  MatSetOptionsPrefix(_matA, options_prefix.c_str());
}
//-----------------------------------------------------------------------------
std::string PETScMatrix::get_options_prefix() const
{
  assert(_matA);
  const char* prefix = nullptr;
  MatGetOptionsPrefix(_matA, &prefix);
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_from_options()
{
  assert(_matA);
  MatSetFromOptions(_matA);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_nullspace(const la::VectorSpaceBasis& nullspace)
{
  assert(_matA);

  // Get matrix communicator
  MPI_Comm comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_matA, &comm);

  // Create PETSc nullspace
  MatNullSpace petsc_ns = create_petsc_nullspace(comm, nullspace);

  // Attach PETSc nullspace to matrix
  assert(_matA);
  PetscErrorCode ierr = MatSetNullSpace(_matA, petsc_ns);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetNullSpace");

  // Decrease reference count for nullspace by destroying
  MatNullSpaceDestroy(&petsc_ns);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_near_nullspace(const la::VectorSpaceBasis& nullspace)
{
  assert(_matA);

  // Get matrix communicator
  MPI_Comm comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_matA, &comm);

  // Create PETSc nullspace
  MatNullSpace petsc_ns = la::create_petsc_nullspace(comm, nullspace);

  // Attach near  nullspace to matrix
  assert(_matA);
  PetscErrorCode ierr = MatSetNearNullSpace(_matA, petsc_ns);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetNullSpace");

  // Decrease reference count for nullspace
  MatNullSpaceDestroy(&petsc_ns);
}
//-----------------------------------------------------------------------------
