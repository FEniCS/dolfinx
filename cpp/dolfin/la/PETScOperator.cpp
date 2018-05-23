// Copyright (C) 2011-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScOperator.h"
#include "PETScVector.h"
#include "utils.h"
#include <dolfin/log/log.h>
#include <petscvec.h>

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
PETScOperator::PETScOperator() : _matA(nullptr) {}
//-----------------------------------------------------------------------------
PETScOperator::PETScOperator(Mat A) : _matA(A)
{
  // Increase reference count, and throw error if Mat pointer is NULL
  if (_matA)
    PetscObjectReference((PetscObject)_matA);
  else
  {
    throw std::runtime_error(
        "Cannot wrap PETSc Mat objects that have not been initialized");
  }
}
//-----------------------------------------------------------------------------
PETScOperator::PETScOperator(PETScOperator&& A) : _matA(nullptr)
{
  _matA = A._matA;
  A._matA = nullptr;
}
//-----------------------------------------------------------------------------
PETScOperator::~PETScOperator()
{
  // Decrease reference count (PETSc will destroy object once
  // reference counts reached zero)
  if (_matA)
    MatDestroy(&_matA);
}
//-----------------------------------------------------------------------------
PETScOperator& PETScOperator::operator=(PETScOperator&& A)
{
  if (_matA)
    MatDestroy(&_matA);
  _matA = A._matA;
  A._matA = nullptr;
  return *this;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScOperator::size() const
{
  assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetSize(_matA, &m, &n);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MetGetSize");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScOperator::local_range(std::size_t dim) const
{
  assert(dim <= 1);
  if (dim == 1)
  {
    throw std::runtime_error(
        "Only local row range is available for PETSc matrices");
  }

  assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetOwnershipRange(_matA, &m, &n);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatGetOwnershipRange");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
PETScVector PETScOperator::init_vector(std::size_t dim) const
{
  assert(_matA);
  PetscErrorCode ierr;

  Vec x = nullptr;
  if (dim == 0)
  {
    ierr = MatCreateVecs(_matA, NULL, &x);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatCreateVecs");
  }
  else if (dim == 1)
  {
    ierr = MatCreateVecs(_matA, &x, NULL);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatCreateVecs");
  }
  else
  {
    log::dolfin_error("PETScOperator.cpp",
                      "initialize PETSc vector to match PETSc matrix",
                      "Dimension must be 0 or 1, not %d", dim);
  }

  // Associate new PETSc Vec with z (this will increase the reference
  // count to x)
  PETScVector z(x);

  // Decrease reference count
  VecDestroy(&x);

  return z;
}
//-----------------------------------------------------------------------------
MPI_Comm PETScOperator::mpi_comm() const
{
  assert(_matA);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_matA, &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------
Mat PETScOperator::mat() const { return _matA; }
//-----------------------------------------------------------------------------
std::string PETScOperator::str(bool verbose) const
{
  return "No str function for this PETSc matrix operator.";
}
//-----------------------------------------------------------------------------
