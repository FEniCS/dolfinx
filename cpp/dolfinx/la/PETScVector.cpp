// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScVector.h"
#include "utils.h"
#include <algorithm>
#include <cstddef>
#include <dolfinx/common/IndexMap.h>
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
std::vector<IS> la::create_petsc_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  std::vector<IS> is;
  std::int64_t offset = 0;
  for (auto& map : maps)
  {
    const int bs = map.second;
    const std::int32_t size
        = map.first.get().size_local() + map.first.get().num_ghosts();
    IS _is;
    ISCreateStride(PETSC_COMM_SELF, bs * size, offset, 1, &_is);
    is.push_back(_is);
    offset += bs * size;
  }

  return is;
}
//-----------------------------------------------------------------------------
Vec la::create_ghosted_vector(const common::IndexMap& map, int bs,
                              tcb::span<PetscScalar> x)
{
  const std::int32_t size_local = bs * map.size_local();
  const std::int64_t size_global = bs * map.size_global();
  const std::vector<PetscInt> ghosts(map.ghosts().begin(), map.ghosts().end());
  Vec vec;
  VecCreateGhostBlockWithArray(map.comm(), bs, size_local, size_global,
                               ghosts.size(), ghosts.data(), x.data(), &vec);
  return vec;
}
//-----------------------------------------------------------------------------
Vec la::create_petsc_vector(const dolfinx::common::IndexMap& map, int bs)
{
  return la::create_petsc_vector(map.comm(), map.local_range(), map.ghosts(),
                                 bs);
}
//-----------------------------------------------------------------------------
Vec la::create_petsc_vector(MPI_Comm comm, std::array<std::int64_t, 2> range,
                            const std::vector<std::int64_t>& ghosts, int bs)
{
  PetscErrorCode ierr;

  // Get local size
  assert(range[1] >= range[0]);
  const std::int32_t local_size = range[1] - range[0];

  Vec x;
  const std::vector<PetscInt> _ghosts(ghosts.begin(), ghosts.end());
  ierr = VecCreateGhostBlock(comm, bs, bs * local_size, PETSC_DETERMINE,
                             _ghosts.size(), _ghosts.data(), &x);
  CHECK_ERROR("VecCreateGhostBlock");
  assert(x);

  return x;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<PetscScalar>> la::get_local_vectors(
    const Vec x,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  // Get ghost offset
  int offset_owned = 0;
  for (auto& map : maps)
    offset_owned += map.first.get().size_local() * map.second;

  // Unwrap PETSc vector
  Vec x_local;
  VecGhostGetLocalForm(x, &x_local);
  PetscInt n = 0;
  VecGetSize(x_local, &n);
  const PetscScalar* array = nullptr;
  VecGetArrayRead(x_local, &array);
  tcb::span _x(array, n);

  // Copy PETSc Vec data in to local vectors
  std::vector<std::vector<PetscScalar>> x_b;
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (auto map : maps)
  {
    const std::int32_t size_owned = map.first.get().size_local() * map.second;
    const std::int32_t size_ghost = map.first.get().num_ghosts() * map.second;

    x_b.emplace_back(size_owned + size_ghost);
    std::copy_n(std::next(_x.begin(), offset), size_owned, x_b.back().begin());
    std::copy_n(std::next(_x.begin(), offset_ghost), size_ghost,
                std::next(x_b.back().begin(), size_owned));

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  VecRestoreArrayRead(x_local, &array);
  VecGhostRestoreLocalForm(x, &x_local);

  return x_b;
}
//-----------------------------------------------------------------------------
void la::scatter_local_vectors(
    Vec x, const std::vector<tcb::span<const PetscScalar>>& x_b,
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  if (x_b.size() != maps.size())
    throw std::runtime_error("Mismatch in vector/map size.");

  // Get ghost offset
  int offset_owned = 0;
  for (auto& map : maps)
    offset_owned += map.first.get().size_local() * map.second;

  Vec x_local;
  VecGhostGetLocalForm(x, &x_local);
  PetscInt n = 0;
  VecGetSize(x_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(x_local, &array);
  tcb::span _x(array, n);

  // Copy local vectors into PETSc Vec
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    const std::int32_t size_owned
        = maps[i].first.get().size_local() * maps[i].second;
    const std::int32_t size_ghost
        = maps[i].first.get().num_ghosts() * maps[i].second;

    std::copy_n(x_b[i].begin(), size_owned, std::next(_x.begin(), offset));
    std::copy_n(std::next(x_b[i].begin(), size_owned), size_ghost,
                std::next(_x.begin(), offset_ghost));

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  VecRestoreArray(x_local, &array);
  VecGhostRestoreLocalForm(x, &x_local);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
PETScVector::PETScVector(const common::IndexMap& map, int bs)
    : _x(la::create_petsc_vector(map, bs))
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
PETScVector::PETScVector(PETScVector&& v) : _x(std::exchange(v._x, nullptr)) {}
//-----------------------------------------------------------------------------
PETScVector::~PETScVector()
{
  if (_x)
    VecDestroy(&_x);
}
//-----------------------------------------------------------------------------
PETScVector& PETScVector::operator=(PETScVector&& v)
{
  std::swap(_x, v._x);
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
MPI_Comm PETScVector::mpi_comm() const
{
  assert(_x);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscErrorCode ierr = PetscObjectGetComm((PetscObject)(_x), &mpi_comm);
  CHECK_ERROR("PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------
PetscReal PETScVector::norm(la::Norm type) const
{
  assert(_x);
  PetscErrorCode ierr;
  PetscReal value = 0.0;
  switch (type)
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
