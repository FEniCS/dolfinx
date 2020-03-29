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
#include <utility>

#include <petsc.h>

// Ceiling division of nonnegative integers
#define dolfin_ceil_div(x, y) (x / y + int(x % y != 0))

#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc_error(ierr, __FILE__, NAME);                                       \
  } while (0)

//-----------------------------------------------------------------------------
Vec dolfinx::la::create_petsc_vector(const dolfinx::common::IndexMap& map)
{
  return dolfinx::la::create_petsc_vector(map.mpi_comm(), map.local_range(),
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
  const std::size_t local_size = range[1] - range[0];

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

  // NOTE: shouldn't need to do this, but there appears to be an issue
  // with PETSc
  // (https://lists.mcs.anl.gov/pipermail/petsc-dev/2018-May/022963.html)
  // Set local-to-global map
  std::vector<PetscInt> l2g(local_size + ghost_indices.size());
  std::iota(l2g.begin(), l2g.begin() + local_size, range[0]);
  std::copy(ghost_indices.data(), ghost_indices.data() + ghost_indices.size(),
            l2g.begin() + local_size);
  ISLocalToGlobalMapping petsc_local_to_global;
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, block_size, l2g.size(),
                                      l2g.data(), PETSC_COPY_VALUES,
                                      &petsc_local_to_global);
  CHECK_ERROR("ISLocalToGlobalMappingCreate");
  ierr = VecSetLocalToGlobalMapping(x, petsc_local_to_global);
  CHECK_ERROR("VecSetLocalToGlobalMapping");
  ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global);
  CHECK_ERROR("ISLocalToGlobalMappingDestroy");

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
      = {{sparsity_pattern.index_map(0), sparsity_pattern.index_map(1)}};
  const int bs0 = index_maps[0]->block_size();
  const int bs1 = index_maps[1]->block_size();

  // Get global and local dimensions
  const std::size_t M = bs0 * index_maps[0]->size_global();
  const std::size_t N = bs1 * index_maps[1]->size_global();
  const std::size_t m = bs0 * index_maps[0]->size_local();
  const std::size_t n = bs1 * index_maps[1]->size_local();

  // Find common block size across rows/columns
  const int bs = (bs0 == bs1 ? bs0 : 1);

  // Set matrix size
  ierr = MatSetSizes(A, m, n, M, N);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetSizes");

  // Get number of nonzeros for each row from sparsity pattern
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> nnz_diag
      = sparsity_pattern.num_nonzeros_diagonal();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> nnz_offdiag
      = sparsity_pattern.num_nonzeros_off_diagonal();

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(A);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetFromOptions");

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _nnz_diag(nnz_diag.size() / bs),
      _nnz_offdiag(nnz_offdiag.size() / bs);

  for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
    _nnz_diag[i] = dolfin_ceil_div(nnz_diag[bs * i], bs);
  for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
    _nnz_offdiag[i] = dolfin_ceil_div(nnz_offdiag[bs * i], bs);

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
    auto *x = nullspace[i]->vec();

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
  std::size_t offset = 0;
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    assert(maps[i]);
    const int size = maps[i]->size_local() + maps[i]->num_ghosts();
    const int bs = maps[i]->block_size();
    std::vector<PetscInt> index(bs * size);
    std::iota(index.begin(), index.end(), offset);

    ISCreateBlock(PETSC_COMM_SELF, 1, index.size(), index.data(),
                  PETSC_COPY_VALUES, &is[i]);
    offset += bs * size;
    // ISCreateBlock(MPI_COMM_SELF, bs, index.size(), index.data(),
    //               PETSC_COPY_VALUES, &is[i]);
    // offset += size;
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
dolfinx::la::VecWrapper::VecWrapper(Vec y, bool ghosted)
    : x(nullptr, 0), _y(y),  _ghosted(ghosted)
{
  assert(_y);
  if (ghosted)
    VecGhostGetLocalForm(_y, &_y_local);
  else
    _y_local = _y;

  PetscInt n = 0;
  VecGetSize(_y_local, &n);
  VecGetArray(_y_local, &array);

  new (&x) Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>(array, n);
}
//-----------------------------------------------------------------------------
dolfinx::la::VecWrapper::VecWrapper(VecWrapper&& w)
    : x(std::move(w.x)), _y(std::exchange(w._y, nullptr)),
      _y_local(std::exchange(w._y_local, nullptr)),
      _ghosted(std::move(w._ghosted))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfinx::la::VecWrapper::~VecWrapper()
{
  if (_y_local)
  {
    VecRestoreArray(_y_local, &array);
    if (_ghosted)
      VecGhostRestoreLocalForm(_y, &_y_local);
  }
}
//-----------------------------------------------------------------------------
dolfinx::la::VecWrapper& dolfinx::la::VecWrapper::operator=(VecWrapper&& w)
{
  _y = std::exchange(w._y, nullptr);
  _y_local = std::exchange(w._y_local, nullptr);
  _ghosted = std::move(w._ghosted);

  return *this;
}
//-----------------------------------------------------------------------------
void dolfinx::la::VecWrapper::restore()
{
  assert(_y);
  assert(_y_local);
  VecRestoreArray(_y_local, &array);
  if (_ghosted)
    VecGhostRestoreLocalForm(_y, &_y_local);

  _y = nullptr;
  _y_local = nullptr;
  new (&x)
      Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>(nullptr, 0);
}
//-----------------------------------------------------------------------------
dolfinx::la::VecReadWrapper::VecReadWrapper(const Vec y, bool ghosted)
    : x(nullptr, 0), _y(y),  _ghosted(ghosted)
{
  assert(_y);
  if (ghosted)
    VecGhostGetLocalForm(_y, &_y_local);
  else
    _y_local = _y;

  PetscInt n = 0;
  VecGetSize(_y_local, &n);
  VecGetArrayRead(_y_local, &array);
  new (&x)
      Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>(array, n);
}
//-----------------------------------------------------------------------------
dolfinx::la::VecReadWrapper::VecReadWrapper(VecReadWrapper&& w)
    : x(std::move(w.x)), _y(std::exchange(w._y, nullptr)),
      _y_local(std::exchange(w._y_local, nullptr)),
      _ghosted(std::move(w._ghosted))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfinx::la::VecReadWrapper::~VecReadWrapper()
{
  if (_y_local)
  {
    VecRestoreArrayRead(_y_local, &array);
    if (_ghosted)
      VecGhostRestoreLocalForm(_y, &_y_local);
  }
}
//-----------------------------------------------------------------------------
dolfinx::la::VecReadWrapper&
dolfinx::la::VecReadWrapper::operator=(VecReadWrapper&& w)
{
  _y = std::exchange(w._y, nullptr);
  _y_local = std::exchange(w._y_local, nullptr);
  _ghosted = std::move(w._ghosted);

  return *this;
}
//-----------------------------------------------------------------------------
void dolfinx::la::VecReadWrapper::restore()
{
  assert(_y);
  assert(_y_local);
  VecRestoreArrayRead(_y_local, &array);
  if (_ghosted)
    VecGhostRestoreLocalForm(_y, &_y_local);

  _y = nullptr;
  _y_local = nullptr;
  new (&x)
      Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>(nullptr, 0);
}
//-----------------------------------------------------------------------------
// Mat dolfinx::la::get_local_submatrix(const Mat A, const IS row, const IS
// col);

// void restore_local_submatrix(const Mat A, const IS row, const IS col, Mat*
// Asub);

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
  la::VecReadWrapper x_wrapper(x);

  // Copy PETSc Vec data in to Eigen vector
  std::vector<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x_b;
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  for (const common::IndexMap* map : maps)
  {
    const int bs = map->block_size();
    const int size_owned = map->size_local() * bs;
    const int size_ghost = map->num_ghosts() * bs;
    x_b.emplace_back(
        Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>(size_owned + size_ghost));
    x_b.back().head(size_owned) = x_wrapper.x.segment(offset, size_owned);
    x_b.back().tail(size_ghost) = x_wrapper.x.segment(offset_ghost, size_ghost);

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  x_wrapper.restore();

  return x_b;
}
//-----------------------------------------------------------------------------
// std::vector<Vec> dolfinx::la::get_local_petsc_vectors(
//     const Vec x, const std::vector<const common::IndexMap*>& maps)
// {
//     // Get ghost offset
//   int offset_owned = 0;
//   for (const common::IndexMap* map : maps)
//   {
//     assert(map);
//     offset_owned += map->size_local() * map->block_size();
//   }

//   // Unwrap PETSc vector
//   la::VecReadWrapper x_wrapper(x);

//   // Copy PETSc Vec data in to Eigen vector
//   std::vector<Vec> x_b;
//   int offset = 0;
//   int offset_ghost = offset_owned; // Ghost DoFs start after owned
//   for (const common::IndexMap* map : maps)
//   {
//     VecCreate

//     const int bs = map->block_size();
//     const int size_owned = map->size_local() * bs;
//     const int size_ghost = map->num_ghosts() * bs;
//     x_b.emplace_back(
//         Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>(size_owned +
//         size_ghost));
//     x_b.back().head(size_owned) = x_wrapper.x.segment(offset, size_owned);
//     x_b.back().tail(size_ghost) = x_wrapper.x.segment(offset_ghost,
//     size_ghost);

//     offset += size_owned;
//     offset_ghost += size_ghost;
//   }

//   x_wrapper.restore();

//   return x_b;

// }
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
  int offset = 0;
  int offset_ghost = offset_owned; // Ghost DoFs start after owned
  la::VecWrapper x_wrapper(x);
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    const int bs = maps[i]->block_size();
    const int size_owned = maps[i]->size_local() * bs;
    const int size_ghost = maps[i]->num_ghosts() * bs;
    x_wrapper.x.segment(offset, size_owned) = x_b[i].head(size_owned);
    x_wrapper.x.segment(offset_ghost, size_ghost) = x_b[i].tail(size_ghost);

    offset += size_owned;
    offset_ghost += size_ghost;
  }

  x_wrapper.restore();
}
//-----------------------------------------------------------------------------
