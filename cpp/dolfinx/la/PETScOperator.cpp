// Copyright (C) 2011-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScOperator.h"
#include "PETScVector.h"
#include <cassert>
#include <dolfinx/common/log.h>
#include <petscvec.h>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
petsc::Operator::Operator(Mat A, bool inc_ref_count) : _matA(A)
{
  assert(A);
  if (inc_ref_count)
    PetscObjectReference((PetscObject)_matA);
}
//-----------------------------------------------------------------------------
petsc::Operator::Operator(Operator&& A) : _matA(std::exchange(A._matA, nullptr))
{
}
//-----------------------------------------------------------------------------
petsc::Operator::~Operator()
{
  // Decrease reference count (PETSc will destroy object once reference
  // counts reached zero)
  if (_matA)
    MatDestroy(&_matA);
}
//-----------------------------------------------------------------------------
petsc::Operator& petsc::Operator::operator=(Operator&& A)
{
  std::swap(_matA, A._matA);
  return *this;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> petsc::Operator::size() const
{
  assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetSize(_matA, &m, &n);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MetGetSize");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
petsc::Vector petsc::Operator::create_vector(std::size_t dim) const
{
  assert(_matA);
  PetscErrorCode ierr;

  Vec x = nullptr;
  if (dim == 0)
  {
    ierr = MatCreateVecs(_matA, nullptr, &x);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatCreateVecs");
  }
  else if (dim == 1)
  {
    ierr = MatCreateVecs(_matA, &x, nullptr);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatCreateVecs");
  }
  else
  {
    LOG(ERROR) << "Cannot initialize PETSc vector to match PETSc matrix. "
               << "Dimension must be 0 or 1, not " << dim;
    throw std::runtime_error("Invalid dimension");
  }

  return Vector(x, false);
}
//-----------------------------------------------------------------------------
Mat petsc::Operator::mat() const { return _matA; }
//-----------------------------------------------------------------------------
