// Copyright (C) 2011-2012 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "PETScBaseMatrix.h"
#include "PETScVector.h"
#include <dolfin/log/log.h>
#include <petscvec.h>

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
PETScBaseMatrix::PETScBaseMatrix(Mat A) : _matA(A)
{
  // Increase reference count, and throw error if Mat pointer is NULL
  if (_matA)
    PetscObjectReference((PetscObject)_matA);
  else
  {
    log::dolfin_error(
        "PETScBaseMatrix.cpp", "initialize with PETSc Mat pointer",
        "Cannot wrap PETSc Mat objects that have not been initialized");
  }
}
//-----------------------------------------------------------------------------
PETScBaseMatrix::~PETScBaseMatrix()
{
  // Decrease reference count (PETSc will destroy object once
  // reference counts reached zero)
  if (_matA)
    MatDestroy(&_matA);
}
//-----------------------------------------------------------------------------
PETScBaseMatrix::PETScBaseMatrix(const PETScBaseMatrix& A)
{
  log::dolfin_error("PETScBaseMatrix.cpp", "copy constructor",
               "PETScBaseMatrix does not provide a copy constructor");
}
//-----------------------------------------------------------------------------
std::int64_t PETScBaseMatrix::size(std::size_t dim) const
{
  if (dim > 1)
  {
    log::dolfin_error("PETScBaseMatrix.cpp", "access size of PETSc matrix",
                 "Illegal axis (%d), must be 0 or 1", dim);
  }

  dolfin_assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetSize(_matA, &m, &n);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MetGetSize");
  if (dim == 0)
    return m > 0 ? m : 0;
  else
    return n > 0 ? n : 0;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScBaseMatrix::size() const
{
  dolfin_assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetSize(_matA, &m, &n);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MetGetSize");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScBaseMatrix::local_range(std::size_t dim) const
{
  dolfin_assert(dim <= 1);
  if (dim == 1)
  {
    log::dolfin_error("PETScBaseMatrix.cpp",
                 "access local column range for PETSc matrix",
                 "Only local row range is available for PETSc matrices");
  }

  dolfin_assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetOwnershipRange(_matA, &m, &n);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatGetOwnershipRange");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
void PETScBaseMatrix::init_vector(PETScVector& z, std::size_t dim) const
{
  dolfin_assert(_matA);
  PetscErrorCode ierr;

  // Create new PETSc vector
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
    log::dolfin_error("PETScBaseMatrix.cpp",
                 "initialize PETSc vector to match PETSc matrix",
                 "Dimension must be 0 or 1, not %d", dim);
  }

  // Associate new PETSc Vec with z (this will increase the reference
  // count to x)
  z.reset(x);

  // Decrease reference count
  VecDestroy(&x);
}
//-----------------------------------------------------------------------------
MPI_Comm PETScBaseMatrix::mpi_comm() const
{
  dolfin_assert(_matA);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_matA, &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------

#endif
