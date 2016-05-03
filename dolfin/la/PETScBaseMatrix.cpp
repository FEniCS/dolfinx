// Copyright (C) 2011-2012 Anders Logg and Garth N. Wells
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
// First added:  2011-02-11
// Last changed: 2012-08-22

#ifdef HAS_PETSC

#include <dolfin/log/log.h>
#include "GenericVector.h"
#include "PETScVector.h"
#include "PETScBaseMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScBaseMatrix::PETScBaseMatrix(Mat A) : _matA(A)
{
  // Increase reference count, and throw error if Mat pointer is NULL
  if (_matA)
    PetscObjectReference((PetscObject)_matA);
  else
  {
    dolfin_error("PETScBaseMatrix.cpp",
                 "intialise with PETSc Mat pointer",
                 "Cannot wrap PETSc Mat objects that have not been intialized");
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
  dolfin_error("PETScBaseMatrix.cpp",
               "copy constructor",
               "PETScBaseMatrix does not provide a copy constructor");
}
//-----------------------------------------------------------------------------
std::size_t PETScBaseMatrix::size(std::size_t dim) const
{
  if (dim > 1)
  {
    dolfin_error("PETScBaseMatrix.cpp",
                 "access size of PETSc matrix",
                 "Illegal axis (%d), must be 0 or 1", dim);
  }

  dolfin_assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetSize(_matA, &m, &n);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MetGetSize");
  if (dim == 0)
    return m > 0 ? m : 0;
  else
    return n > 0 ? n : 0;
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::int64_t> PETScBaseMatrix::size() const
{
  dolfin_assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetSize(_matA, &m, &n);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MetGetSize");
  return {m, n};
}
//-----------------------------------------------------------------------------
std::pair<std::int64_t, std::int64_t>
PETScBaseMatrix::local_range(std::size_t dim) const
{
  dolfin_assert(dim <= 1);
  if (dim == 1)
  {
    dolfin_error("PETScBaseMatrix.cpp",
                 "access local column range for PETSc matrix",
                 "Only local row range is available for PETSc matrices");
  }

  dolfin_assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetOwnershipRange(_matA, &m, &n);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetOwnershipRange");
  return {m, n};
}
//-----------------------------------------------------------------------------
void PETScBaseMatrix::init_vector(GenericVector& z, std::size_t dim) const
{
  dolfin_assert(_matA);
  PetscErrorCode ierr;

  // Downcast vector
  PETScVector& _z = as_type<PETScVector>(z);

  // Create new PETSc vector
  Vec x = nullptr;
  if (dim == 0)
  {
    ierr = MatCreateVecs(_matA, NULL, &x);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatCreateVecs");
  }
  else if (dim == 1)
  {
    ierr = MatCreateVecs(_matA, &x, NULL);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatCreateVecs");
  }
  else
  {
    dolfin_error("PETScBaseMatrix.cpp",
                 "initialize PETSc vector to match PETSc matrix",
                 "Dimension must be 0 or 1, not %d", dim);
  }

  // Associate new PETSc Vec with _z
  _z._x = x;
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
