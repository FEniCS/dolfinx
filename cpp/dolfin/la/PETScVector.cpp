// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScVector.h"
#include "utils.h"
#include <cstddef>
#include <cstring>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/Timer.h>

using namespace dolfin;
using namespace dolfin::la;

#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc_error(ierr, __FILE__, NAME);                                       \
  } while (0)

//-----------------------------------------------------------------------------
PETScVector::PETScVector(const common::IndexMap& map)
    : _x(la::create_petsc_vector(map))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
    int block_size)
    : _x(la::create_petsc_vector(comm, range, ghost_indices, block_size))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(Vec x, bool inc_ref_count) : _x(x)
{
  assert(x);
  if (inc_ref_count)
    PetscObjectReference((PetscObject)_x);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(PETScVector&& v) : _x(v._x) { v._x = nullptr; }
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  if (_x)
    VecDestroy(&_x);
}
//-----------------------------------------------------------------------------
PETScVector& PETScVector::operator=(PETScVector&& v)
{
  if (_x)
    VecDestroy(&_x);
  _x = v._x;
  v._x = nullptr;
  return *this;
}
//-----------------------------------------------------------------------------
PETScVector PETScVector::copy() const
{
  Vec _y;
  VecDuplicate(_x, &_y);
  VecCopy(_x, _y);
  PETScVector y(_y);
  VecDestroy(&_y);
  return y;
}
//-----------------------------------------------------------------------------
std::int64_t PETScVector::size() const
{
  assert(_x);
  PetscInt n = 0;
  PetscErrorCode ierr = VecGetSize(_x, &n);
  CHECK_ERROR("VecGetSize");
  return n;
}
//-----------------------------------------------------------------------------
std::size_t PETScVector::local_size() const
{
  assert(_x);
  PetscInt n = 0;
  PetscErrorCode ierr = VecGetLocalSize(_x, &n);
  CHECK_ERROR("VecGetLocalSize");
  return n;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScVector::local_range() const
{
  assert(_x);
  PetscInt n0, n1;
  PetscErrorCode ierr = VecGetOwnershipRange(_x, &n0, &n1);
  CHECK_ERROR("VecGetOwnershipRange");
  assert(n0 <= n1);
  return {{n0, n1}};
}
//-----------------------------------------------------------------------------
void PETScVector::apply_ghosts()
{
  assert(_x);
  PetscErrorCode ierr;

  Vec xg;
  ierr = VecGhostGetLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostGetLocalForm");
  if (xg) // Vec is ghosted
  {
    ierr = VecGhostUpdateBegin(_x, ADD_VALUES, SCATTER_REVERSE);
    CHECK_ERROR("VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(_x, ADD_VALUES, SCATTER_REVERSE);
    CHECK_ERROR("VecGhostUpdateEnd");
  }

  ierr = VecGhostRestoreLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
void PETScVector::update_ghosts()
{
  assert(_x);
  PetscErrorCode ierr;

  Vec xg;
  ierr = VecGhostGetLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostGetLocalForm");
  if (xg) // Vec is ghosted
  {
    ierr = VecGhostUpdateBegin(_x, INSERT_VALUES, SCATTER_FORWARD);
    CHECK_ERROR("VecGhostUpdateBegin");
    ierr = VecGhostUpdateEnd(_x, INSERT_VALUES, SCATTER_FORWARD);
    CHECK_ERROR("VecGhostUpdateEnd");
  }

  ierr = VecGhostRestoreLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostRestoreLocalForm");
}
//-----------------------------------------------------------------------------
MPI_Comm PETScVector::mpi_comm() const
{
  assert(_x);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscErrorCode ierr = PetscObjectGetComm((PetscObject)(_x), &mpi_comm);
  CHECK_ERROR("PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------
PetscReal PETScVector::norm(la::Norm norm_type) const
{
  assert(_x);
  PetscErrorCode ierr;
  PetscReal value = 0.0;
  switch (norm_type)
  {
  case la::Norm::l1:
    ierr = VecNorm(_x, NORM_1, &value);
    CHECK_ERROR("VecNorm");
    break;
  case la::Norm::l2:
    ierr = VecNorm(_x, NORM_2, &value);
    CHECK_ERROR("VecNorm");
    break;
  case la::Norm::linf:
    ierr = VecNorm(_x, NORM_INFINITY, &value);
    CHECK_ERROR("VecNorm");
    break;
  default:
    throw std::runtime_error("Norm type not support for PETSc Vec");
  }

  return value;
}
//-----------------------------------------------------------------------------
void PETScVector::set_options_prefix(std::string options_prefix)
{
  assert(_x);
  PetscErrorCode ierr = VecSetOptionsPrefix(_x, options_prefix.c_str());
  CHECK_ERROR("VecSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string PETScVector::get_options_prefix() const
{
  assert(_x);
  const char* prefix = nullptr;
  PetscErrorCode ierr = VecGetOptionsPrefix(_x, &prefix);
  CHECK_ERROR("VecGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void PETScVector::set_from_options()
{
  assert(_x);
  PetscErrorCode ierr = VecSetFromOptions(_x);
  CHECK_ERROR("VecSetFromOptions");
}
//-----------------------------------------------------------------------------
Vec PETScVector::vec() const { return _x; }
//-----------------------------------------------------------------------------
