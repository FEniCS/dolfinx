// Copyright (C) 2013-2019 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "PETScVector.h"
#include "VectorSpaceBasis.h"
#include <cassert>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/la/SparsityPattern.h>
#include <memory>
#include <spdlog/spdlog.h>

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
Vec dolfin::la::create_petsc_vector(const dolfin::common::IndexMap& map)
{
  return dolfin::la::create_petsc_vector(map.mpi_comm(), map.local_range(),
                                         map.ghosts(), map.block_size());
}
//-----------------------------------------------------------------------------
Vec dolfin::la::create_petsc_vector(
    MPI_Comm comm, std::array<std::int64_t, 2> range,
    const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& ghost_indices,
    int block_size)
{
  PetscErrorCode ierr;

  // Get local size
  assert(range[1] >= range[0]);
  const std::size_t local_size = range[1] - range[0];

  Vec x;
  ierr = VecCreateGhostBlock(comm, block_size, block_size * local_size,
                             PETSC_DECIDE, ghost_indices.size(),
                             ghost_indices.data(), &x);
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
Mat dolfin::la::create_petsc_matrix(
    MPI_Comm comm, const dolfin::la::SparsityPattern& sparsity_pattern)
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
  const int bs_map = bs;

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
                                 NULL, NULL);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatXIJSetPreallocation");

  // Build and set local-to-global maps
  assert(bs0 % bs_map == 0);
  assert(bs1 % bs_map == 0);
  std::vector<PetscInt> map0((m + index_maps[0]->num_ghosts())
                             * (bs0 / bs_map));
  std::vector<PetscInt> map1((n + index_maps[1]->num_ghosts())
                             * (bs1 / bs_map));

  const int row_block_size
      = index_maps[0]->size_local() + index_maps[0]->num_ghosts();
  for (int i = 0; i < row_block_size; ++i)
  {
    std::size_t factor = bs0 / bs_map;
    auto index = index_maps[0]->local_to_global(i);
    for (std::size_t j = 0; j < factor; ++j)
      map0[i * factor + j] = factor * index + j;
  }

  const int col_block_size
      = index_maps[1]->size_local() + index_maps[1]->num_ghosts();
  for (int i = 0; i < col_block_size; ++i)
  {
    std::size_t factor = bs1 / bs_map;
    auto index = index_maps[1]->local_to_global(i);
    for (std::size_t j = 0; j < factor; ++j)
      map1[i * factor + j] = factor * index + j;
  }

  // FIXME: In many cases the rows and columns could shared a common
  // local-to-global map

  // Create PETSc local-to-global map/index set
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs_map, map0.size(),
                                      map0.data(), PETSC_COPY_VALUES,
                                      &petsc_local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs_map, map1.size(),
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
MatNullSpace dolfin::la::create_petsc_nullspace(
    MPI_Comm comm, const dolfin::la::VectorSpaceBasis& nullspace)
{
  PetscErrorCode ierr;

  // Copy vectors in vector space object
  std::vector<Vec> _nullspace;
  for (std::size_t i = 0; i < nullspace.dim(); ++i)
  {
    assert(nullspace[i]);
    auto x = nullspace[i]->vec();

    // Copy vector pointer
    assert(x);
    _nullspace.push_back(x);
  }

  // Create PETSC nullspace
  MatNullSpace petsc_nullspace = NULL;
  ierr = MatNullSpaceCreate(comm, PETSC_FALSE, _nullspace.size(),
                            _nullspace.data(), &petsc_nullspace);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatNullSpaceCreate");

  return petsc_nullspace;
}
//-----------------------------------------------------------------------------
std::vector<IS> dolfin::la::compute_petsc_index_sets(
    std::vector<const dolfin::common::IndexMap*> maps)
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
    ISCreateBlock(MPI_COMM_SELF, 1, index.size(), index.data(),
                  PETSC_COPY_VALUES, &is[i]);
    // ISCreateBlock(MPI_COMM_SELF, bs, index.size(), index.data(),
    //               PETSC_COPY_VALUES, &is[i]);
    offset += bs * size;
    // offset += size;
  }

  return is;
}
//-----------------------------------------------------------------------------
void dolfin::la::petsc_error(int error_code, std::string filename,
                             std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Fetch and clear PETSc error message
  const std::string msg = common::SubSystemsManager::singleton().petsc_err_msg;
  dolfin::common::SubSystemsManager::singleton().petsc_err_msg = "";

  // Log detailed error info
  spdlog::debug("PETSc error in '%s', '%s'", filename.c_str(),
                petsc_function.c_str());
  spdlog::debug("PETSc error code '%d' (%s), message follows:", error_code,
                desc);
  // NOTE: don't put msg as variadic argument; it might get trimmed
  spdlog::debug(std::string(78, '-'));
  spdlog::debug(msg);
  spdlog::debug(std::string(78, '-'));

  // Raise exception with standard error message
  throw std::runtime_error("Failed to successfully call PETSc function '"
                           + petsc_function + "'. PETSc error code is: "
                           + std ::to_string(error_code) + ", "
                           + std::string(desc));
}
//-----------------------------------------------------------------------------
dolfin::la::VecWrapper::VecWrapper(Vec y, bool ghosted)
    : x(nullptr, 0), _y(y), _y_local(nullptr), _ghosted(ghosted)
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
dolfin::la::VecWrapper::~VecWrapper()
{
  if (_y_local)
  {
    VecRestoreArray(_y_local, &array);
    if (_ghosted)
      VecGhostRestoreLocalForm(_y, &_y_local);
  }
}
//-----------------------------------------------------------------------------
void dolfin::la::VecWrapper::restore()
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
dolfin::la::VecReadWrapper::VecReadWrapper(const Vec y, bool ghosted)
    : x(nullptr, 0), _y(y), _y_local(nullptr), _ghosted(ghosted)
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
dolfin::la::VecReadWrapper::~VecReadWrapper()
{
  if (_y_local)
  {
    VecRestoreArrayRead(_y_local, &array);
    if (_ghosted)
      VecGhostRestoreLocalForm(_y, &_y_local);
  }
}
//-----------------------------------------------------------------------------
void dolfin::la::VecReadWrapper::restore()
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
