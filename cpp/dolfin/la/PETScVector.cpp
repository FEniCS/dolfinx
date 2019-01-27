// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScVector.h"
#include "SparsityPattern.h"
#include "utils.h"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <numeric>

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
    : PETScVector(map.mpi_comm(), map.local_range(), map.ghosts(),
                  map.block_size())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
    int block_size)
    : _x(nullptr)
{
  _x = la::create_vector(comm, range, ghost_indices, block_size);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector() : _x(nullptr) {}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(Vec x) : _x(x)
{
  // Increase reference count to PETSc object
  assert(x);
  PetscObjectReference((PetscObject)_x);
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const PETScVector& v) : _x(nullptr)
{
  PetscErrorCode ierr;
  assert(v._x);
  ierr = VecDuplicate(v._x, &_x);
  CHECK_ERROR("VecDuplicate");
  ierr = VecCopy(v._x, _x);
  CHECK_ERROR("VecCopy");
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
std::int64_t PETScVector::size() const
{
  if (!_x)
  {
    throw std::runtime_error(
        "PETSc vector has not been initialised. Cannot return size.");
  }

  PetscInt n = 0;
  PetscErrorCode ierr = VecGetSize(_x, &n);
  CHECK_ERROR("VecGetSize");
  return n;
}
//-----------------------------------------------------------------------------
std::size_t PETScVector::local_size() const
{
  if (!_x)
  {
    throw std::runtime_error(
        "PETSc vector has not been initialised. Cannot return local size.");
  }

  PetscInt n = 0;
  PetscErrorCode ierr = VecGetLocalSize(_x, &n);
  CHECK_ERROR("VecGetLocalSize");
  return n;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScVector::local_range() const
{
  if (!_x)
  {
    throw std::runtime_error(
        "PETSc vector has not been initialised. Cannot return local range.");
  }

  PetscInt n0, n1;
  PetscErrorCode ierr = VecGetOwnershipRange(_x, &n0, &n1);
  CHECK_ERROR("VecGetOwnershipRange");
  assert(n0 <= n1);
  return {{n0, n1}};
}
//-----------------------------------------------------------------------------
void PETScVector::get_local(PetscScalar* block, std::size_t m,
                            const PetscInt* rows) const
{
  if (m == 0)
    return;

  assert(_x);
  PetscErrorCode ierr;

  // Get ghost vector
  Vec xg = nullptr;
  ierr = VecGhostGetLocalForm(_x, &xg);
  CHECK_ERROR("VecGhostGetLocalForm");

  // Use array access if no ghost points, otherwise use VecGetValues on
  // local ghosted form of vector
  if (!xg)
  {
    // Get pointer to PETSc vector data
    const PetscScalar* data;
    ierr = VecGetArrayRead(_x, &data);
    CHECK_ERROR("VecGetArrayRead");

    for (std::size_t i = 0; i < m; ++i)
      block[i] = data[rows[i]];

    // Restore array
    ierr = VecRestoreArrayRead(_x, &data);
    CHECK_ERROR("VecRestoreArrayRead");
  }
  else
  {
    assert(xg);
    ierr = VecGetValues(xg, m, rows, block);
    CHECK_ERROR("VecGetValues");

    ierr = VecGhostRestoreLocalForm(_x, &xg);
    CHECK_ERROR("VecGhostRestoreLocalForm");
  }
}
//-----------------------------------------------------------------------------
void PETScVector::set(const PetscScalar* block, std::size_t m,
                      const PetscInt* rows)
{
  assert(_x);
  PetscErrorCode ierr = VecSetValues(_x, m, rows, block, INSERT_VALUES);
  CHECK_ERROR("VecSetValues");
}
//-----------------------------------------------------------------------------
void PETScVector::add_local(const PetscScalar* block, std::size_t m,
                            const PetscInt* rows)
{
  assert(_x);
  PetscErrorCode ierr = VecSetValuesLocal(_x, m, rows, block, ADD_VALUES);
  CHECK_ERROR("VecSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScVector::apply()
{
  common::Timer timer("Apply (PETScVector)");
  assert(_x);
  PetscErrorCode ierr;
  ierr = VecAssemblyBegin(_x);
  CHECK_ERROR("VecAssemblyBegin");
  ierr = VecAssemblyEnd(_x);
  CHECK_ERROR("VecAssemblyEnd");

  // Update any ghost values
  update_ghosts();
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
void PETScVector::set(PetscScalar a)
{
  assert(_x);
  PetscErrorCode ierr = VecSet(_x, a);
  CHECK_ERROR("VecSet");
}
//-----------------------------------------------------------------------------
bool PETScVector::empty() const { return _x == nullptr ? true : false; }
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
  if (!_x)
  {
    throw std::runtime_error(
        "Cannot set options prefix. PETSc Vec has not been initialized.");
  }

  // Set PETSc options prefix
  PetscErrorCode ierr = VecSetOptionsPrefix(_x, options_prefix.c_str());
  CHECK_ERROR("VecSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string PETScVector::get_options_prefix() const
{
  if (!_x)
  {
    throw std::runtime_error(
        "Cannot get options prefix. PETSc Vec has not been initialized.");
  }

  const char* prefix = nullptr;
  PetscErrorCode ierr = VecGetOptionsPrefix(_x, &prefix);
  CHECK_ERROR("VecGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void PETScVector::set_from_options()
{
  if (!_x)
  {
    throw std::runtime_error(
        "Cannot call VecSetFromOptions. PETSc Vec has not been initialized.");
  }

  PetscErrorCode ierr = VecSetFromOptions(_x);
  CHECK_ERROR("VecSetFromOptions");
}
//-----------------------------------------------------------------------------
Vec PETScVector::vec() const { return _x; }
//-----------------------------------------------------------------------------
