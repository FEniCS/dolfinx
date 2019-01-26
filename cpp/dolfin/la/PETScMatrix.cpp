// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScMatrix.h"
#include "PETScVector.h"
#include "SparsityPattern.h"
#include "VectorSpaceBasis.h"
#include "utils.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

// Ceiling division of nonnegative integers
#define dolfin_ceil_div(x, y) (x / y + int(x % y != 0))

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix() : PETScOperator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern)
{
  _matA = create_matrix(comm, sparsity_pattern);
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(Mat A) : PETScOperator(A)
{
  // Reference count to A is incremented in base class
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const PETScMatrix& A) : PETScOperator()
{
  assert(A.mat());
  if (!A.empty())
  {
    PetscErrorCode ierr = MatDuplicate(A.mat(), MAT_COPY_VALUES, &_matA);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatDuplicate");
  }
  else
  {
    // Create uninitialised matrix
    PetscErrorCode ierr = MatCreate(A.mpi_comm(), &_matA);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatCreate");
  }
}
//-----------------------------------------------------------------------------
PETScMatrix::~PETScMatrix()
{
  // Do nothing (PETSc matrix is destroyed in base class)
}
//-----------------------------------------------------------------------------
bool PETScMatrix::empty() const { return _matA == nullptr ? true : false; }
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScMatrix::local_range(std::size_t dim) const
{
  return PETScOperator::local_range(dim);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const PetscScalar* block, std::size_t m,
                      const PetscInt* rows, std::size_t n, const PetscInt* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValues(_matA, m, rows, n, cols, block, INSERT_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_local(const PetscScalar* block, std::size_t m,
                            const PetscInt* rows, std::size_t n,
                            const PetscInt* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValuesLocal(_matA, m, rows, n, cols, block, INSERT_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add_local(const PetscScalar* block, std::size_t m,
                            const PetscInt* rows, std::size_t n,
                            const PetscInt* cols)
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
void PETScMatrix::zero()
{
  assert(_matA);
  PetscErrorCode ierr = MatZeroEntries(_matA);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatZeroEntries");
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
  const char* prefix = NULL;
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
  // Create PETSc nullspace
  MatNullSpace petsc_ns = create_petsc_nullspace(mpi_comm(), nullspace);

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
  // Create PETSc nullspace
  MatNullSpace petsc_ns = la::create_petsc_nullspace(mpi_comm(), nullspace);

  // Attach near  nullspace to matrix
  assert(_matA);
  PetscErrorCode ierr = MatSetNearNullSpace(_matA, petsc_ns);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetNullSpace");

  // Decrease reference count for nullspace
  MatNullSpaceDestroy(&petsc_ns);
}
//-----------------------------------------------------------------------------
std::string PETScMatrix::str(bool verbose) const
{
  assert(_matA);
  if (this->empty())
    return "<Uninitialized PETScMatrix>";

  std::stringstream s;
  if (verbose)
  {
    log::warning(
        "Verbose output for PETScMatrix not implemented, calling PETSc "
        "MatView directly.");

    // FIXME: Maybe this could be an option?
    assert(_matA);
    PetscErrorCode ierr;
    if (MPI::size(mpi_comm()) > 1)
    {
      ierr = MatView(_matA, PETSC_VIEWER_STDOUT_WORLD);
      if (ierr != 0)
        petsc_error(ierr, __FILE__, "MatView");
    }
    else
    {
      ierr = MatView(_matA, PETSC_VIEWER_STDOUT_SELF);
      if (ierr != 0)
        petsc_error(ierr, __FILE__, "MatView");
    }
  }
  else
  {
    const std::array<std::int64_t, 2> size = this->size();
    s << "<PETScMatrix of size " << size[0] << " x " << size[1] << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
