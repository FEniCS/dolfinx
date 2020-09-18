// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScVector.h"
#include "utils.h"
#include <cstddef>
#include <cstring>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>

using namespace dolfinx;
using namespace dolfinx::la;

#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc_error(ierr, __FILE__, NAME);                                       \
  } while (0)

//-----------------------------------------------------------------------------
void la::petsc_error(int error_code, std::string filename,
                     std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Log detailed error info
  DLOG(INFO) << "PETSc error in '" << filename.c_str() << "', '"
             << petsc_function.c_str() << "'";
  DLOG(INFO) << "PETSc error code '" << error_code << "' (" << desc << ".";

  throw std::runtime_error("Failed to successfully call PETSc function '"
                           + petsc_function + "'. PETSc error code is: "
                           + std ::to_string(error_code) + ", "
                           + std::string(desc));
}
//-----------------------------------------------------------------------------
std::vector<IS>
la::create_petsc_index_sets(const std::vector<const common::IndexMap*>& maps)
{
  std::vector<IS> is(maps.size());
  std::int64_t offset = 0;
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    assert(maps[i]);
    const std::int32_t size = maps[i]->size_local() + maps[i]->num_ghosts();
    const int bs = maps[i]->block_size();
    std::vector<PetscInt> index(bs * size);
    std::iota(index.begin(), index.end(), offset);

    ISCreateBlock(PETSC_COMM_SELF, 1, index.size(), index.data(),
                  PETSC_COPY_VALUES, &is[i]);
    offset += bs * size;
  }

  return is;
}
//-----------------------------------------------------------------------------
Vec la::create_ghosted_vector(
    const common::IndexMap& map,
    const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>& x)
{
  const int bs = map.block_size();
  std::int32_t size_local = bs * map.size_local();
  std::int32_t num_ghosts = bs * map.num_ghosts();
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts = map.ghosts();
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> _ghosts(bs * ghosts.rows());
  for (int i = 0; i < ghosts.rows(); ++i)
  {
    for (int j = 0; j < bs; ++j)
      _ghosts[i * bs + j] = bs * ghosts[i] + j;
  }

  Vec vec;
  VecCreateGhostWithArray(map.comm(), size_local, PETSC_DECIDE, num_ghosts,
                          _ghosts.data(), x.array().data(), &vec);
  return vec;
}
//-----------------------------------------------------------------------------
Vec la::create_petsc_vector(const dolfinx::common::IndexMap& map)
{
  return la::create_petsc_vector(map.comm(), map.local_range(), map.ghosts(),
                                 map.block_size());
}
//-----------------------------------------------------------------------------
Vec la::create_petsc_vector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>&
        ghost_indices,
    int block_size)
{
  PetscErrorCode ierr;

  // Get local size
  assert(range[1] >= range[0]);
  const std::int32_t local_size = range[1] - range[0];

  Vec x;
  std::vector<PetscInt> _ghost_indices(ghost_indices.rows());
  for (std::size_t i = 0; i < _ghost_indices.size(); ++i)
    _ghost_indices[i] = ghost_indices(i);
  ierr = VecCreateGhostBlock(comm, block_size, block_size * local_size,
                             PETSC_DECIDE, _ghost_indices.size(),
                             _ghost_indices.data(), &x);
  CHECK_ERROR("VecCreateGhostBlock");
  assert(x);

  // Set from PETSc options. This will set the vector type.
  // ierr = VecSetFromOptions(_x);
  // CHECK_ERROR("VecSetFromOptions");

  return x;
}
//-----------------------------------------------------------------------------
std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
la::get_local_vectors(const Vec x,
                      const std::vector<const common::IndexMap*>& maps)
{
  // Get ghost offset
  int offset_owned = 0;
  for (const common::IndexMap* map : maps)
  {
    assert(map);
    offset_owned += map->size_local() * map->block_size();
  }

  // Unwrap PETSc vector
  Vec x_local;
  VecGhostGetLocalForm(x, &x_local);
  PetscInt n = 0;
  VecGetSize(x_local, &n);
  const PetscScalar* array = nullptr;
  VecGetArrayRead(x_local, &array);
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _x(array, n);

  // Copy PETSc Vec data in to Eigen vector
  std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x_b;
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (const common::IndexMap* map : maps)
  {
    const int bs = map->block_size();
    const std::int32_t size_owned = map->size_local() * bs;
    const std::int32_t size_ghost = map->num_ghosts() * bs;
    x_b.emplace_back(
        Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>(size_owned + size_ghost));
    x_b.back().head(size_owned) = _x.segment(offset, size_owned);
    x_b.back().tail(size_ghost) = _x.segment(offset_ghost, size_ghost);

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  VecRestoreArrayRead(x_local, &array);
  VecGhostRestoreLocalForm(x, &x_local);

  return x_b;
}
//-----------------------------------------------------------------------------
void la::scatter_local_vectors(
    Vec x,
    const std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x_b,
    const std::vector<const common::IndexMap*>& maps)
{
  if (x_b.size() != maps.size())
    throw std::runtime_error("Mismatch in vector/map size.");

  // Get ghost offset
  int offset_owned = 0;
  for (const common::IndexMap* map : maps)
  {
    assert(map);
    offset_owned += map->size_local() * map->block_size();
  }

  // Copy Eigen vectors into PETSc Vec
  Vec x_local;
  VecGhostGetLocalForm(x, &x_local);
  PetscInt n = 0;
  VecGetSize(x_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(x_local, &array);
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _x(array, n);

  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    const int bs = maps[i]->block_size();
    const int size_owned = maps[i]->size_local() * bs;
    const int size_ghost = maps[i]->num_ghosts() * bs;
    _x.segment(offset, size_owned) = x_b[i].head(size_owned);
    _x.segment(offset_ghost, size_ghost) = x_b[i].tail(size_ghost);

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  VecRestoreArray(x_local, &array);
  VecGhostRestoreLocalForm(x, &x_local);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const common::IndexMap& map)
    : _x(la::create_petsc_vector(map))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScVector::PETScVector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghost_indices,
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
PETScVector::PETScVector(PETScVector&& v) noexcept : _x(v._x)
{
  v._x = nullptr;
}
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
  PETScVector y(_y, true);
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
std::int32_t PETScVector::local_size() const
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
