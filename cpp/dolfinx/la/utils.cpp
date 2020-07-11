// Copyright (C) 2013-2019 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "PETScVector.h"
#include "VectorSpaceBasis.h"
#include <cassert>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/SubSystemsManager.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/SparsityPattern.h>
#include <memory>
#include <petsc.h>
#include <utility>

#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc_error(ierr, __FILE__, NAME);                                       \
  } while (0)

//-----------------------------------------------------------------------------
Vec dolfinx::la::create_petsc_vector(const dolfinx::common::IndexMap& map)
{
  return dolfinx::la::create_petsc_vector(map.comm(), map.local_range(),
                                          map.ghosts(), map.block_size());
}
//-----------------------------------------------------------------------------
Vec dolfinx::la::create_petsc_vector(
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
Mat dolfinx::la::create_petsc_matrix(
    MPI_Comm comm, const dolfinx::la::SparsityPattern& sparsity_pattern)
{
  PetscErrorCode ierr;
  Mat A;
  ierr = MatCreate(comm, &A);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatCreate");

  // Get IndexMaps from sparsity patterm, and block size
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {sparsity_pattern.index_map(0), sparsity_pattern.index_map(1)};
  const int bs0 = index_maps[0]->block_size();
  const int bs1 = index_maps[1]->block_size();

  // Get global and local dimensions
  const std::int64_t M = bs0 * index_maps[0]->size_global();
  const std::int64_t N = bs1 * index_maps[1]->size_global();
  const std::int32_t m = bs0 * index_maps[0]->size_local();
  const std::int32_t n = bs1 * index_maps[1]->size_local();

  // Find common block size across rows/columns
  const int bs = (bs0 == bs1 ? bs0 : 1);

  // Set matrix size
  ierr = MatSetSizes(A, m, n, M, N);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetSizes");

  // Get number of nonzeros for each row from sparsity pattern
  const graph::AdjacencyList<std::int32_t>& diagonal_pattern
      = sparsity_pattern.diagonal_pattern();
  const graph::AdjacencyList<std::int64_t>& off_diagonal_pattern
      = sparsity_pattern.off_diagonal_pattern();

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(A);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetFromOptions");

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _nnz_diag(index_maps[0]->size_local() * bs0 / bs),
      _nnz_offdiag(index_maps[0]->size_local() * bs0 / bs);

  for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
    _nnz_diag[i] = diagonal_pattern.links(bs * i).rows() / bs;
  for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
    _nnz_offdiag[i] = off_diagonal_pattern.links(bs * i).rows() / bs;

  // Allocate space for matrix
  ierr = MatXAIJSetPreallocation(A, bs, _nnz_diag.data(), _nnz_offdiag.data(),
                                 nullptr, nullptr);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatXIJSetPreallocation");

  // FIXME: In many cases the rows and columns could shared a common
  // local-to-global map

  // Create PETSc local-to-global map/index set
  const bool blocked = (bs0 == bs1 ? true : false);
  const std::vector<std::int64_t> _map0
      = index_maps[0]->global_indices(blocked);
  const std::vector<std::int64_t> _map1
      = index_maps[1]->global_indices(blocked);
  const std::vector<PetscInt> map0(_map0.begin(), _map0.end());
  const std::vector<PetscInt> map1(_map1.begin(), _map1.end());

  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs, map0.size(),
                                      map0.data(), PETSC_COPY_VALUES,
                                      &petsc_local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs, map1.size(),
                                      map1.data(), PETSC_COPY_VALUES,
                                      &petsc_local_to_global1);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Set matrix local-to-global maps
  ierr = MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                                    petsc_local_to_global1);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMappingXXX");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if A is not of type MATIS)
  ierr = MatISSetPreallocation(A, 0, _nnz_diag.data(), 0, _nnz_offdiag.data());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatISSetPreallocation");

  // Clean up local-to-global maps
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Set some options on Mat object
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetOption");
  ierr = MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetOption");

  return A;
}
//-----------------------------------------------------------------------------
MatNullSpace dolfinx::la::create_petsc_nullspace(
    MPI_Comm comm, const dolfinx::la::VectorSpaceBasis& nullspace)
{
  PetscErrorCode ierr;

  // Copy vectors in vector space object
  std::vector<Vec> _nullspace;
  for (int i = 0; i < nullspace.dim(); ++i)
  {
    assert(nullspace[i]);
    Vec x = nullspace[i]->vec();

    // Copy vector pointer
    assert(x);
    _nullspace.push_back(x);
  }

  // Create PETSC nullspace
  MatNullSpace petsc_nullspace = nullptr;
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, _nullspace.size(),
                            _nullspace.data(), &petsc_nullspace);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatNullSpaceCreate");

  return petsc_nullspace;
}
//-----------------------------------------------------------------------------
std::vector<IS> dolfinx::la::create_petsc_index_sets(
    const std::vector<const common::IndexMap*>& maps)
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
void dolfinx::la::petsc_error(int error_code, std::string filename,
                              std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Fetch and clear PETSc error message
  const std::string msg = common::SubSystemsManager::singleton().petsc_err_msg;
  dolfinx::common::SubSystemsManager::singleton().petsc_err_msg = "";

  // // Log detailed error info
  DLOG(INFO) << "PETSc error in '" << filename.c_str() << "', '"
             << petsc_function.c_str() << "'";

  DLOG(INFO) << "PETSc error code '" << error_code << "' (" << desc
             << "), message follows:";

  // NOTE: don't put msg as variadic argument; it might get trimmed
  DLOG(INFO) << std::string(78, '-');
  DLOG(INFO) << msg;
  DLOG(INFO) << std::string(78, '-');

  // Raise exception with standard error message
  throw std::runtime_error("Failed to successfully call PETSc function '"
                           + petsc_function + "'. PETSc error code is: "
                           + std ::to_string(error_code) + ", "
                           + std::string(desc));
}
//-----------------------------------------------------------------------------
std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
dolfinx::la::get_local_vectors(const Vec x,
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
void dolfinx::la::scatter_local_vectors(
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
