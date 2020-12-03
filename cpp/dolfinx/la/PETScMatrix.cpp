// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScMatrix.h"
#include "PETScVector.h"
#include "VectorSpaceBasis.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/SparsityPattern.h>
#include <iostream>
#include <sstream>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
Mat la::create_petsc_matrix(
    MPI_Comm comm, const dolfinx::la::SparsityPattern& sparsity_pattern)
{
  PetscErrorCode ierr;
  Mat A;
  ierr = MatCreate(comm, &A);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatCreate");

  // Get IndexMaps from sparsity patterm, and block size
  std::array maps{sparsity_pattern.index_map(0), sparsity_pattern.index_map(1)};
  const std::array bs
      = {sparsity_pattern.block_size(0), sparsity_pattern.block_size(1)};

  // Get global and local dimensions
  const std::int64_t M = bs[0] * maps[0]->size_global();
  const std::int64_t N = bs[1] * maps[1]->size_global();
  const std::int32_t m = bs[0] * maps[0]->size_local();
  const std::int32_t n = bs[1] * maps[1]->size_local();

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

  // Find a common block size across rows/columns
  const int _bs = (bs[0] == bs[1] ? bs[0] : 1);

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _nnz_diag, _nnz_offdiag;
  if (bs[0] == bs[1])
  {
    _nnz_diag.resize(maps[0]->size_local());
    _nnz_offdiag.resize(maps[0]->size_local());
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = diagonal_pattern.links(i).rows();
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = off_diagonal_pattern.links(i).rows();
  }
  else
  {
    // Expand for block size 1
    _nnz_diag.resize(maps[0]->size_local() * bs[0]);
    _nnz_offdiag.resize(maps[0]->size_local() * bs[0]);
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = bs[1] * diagonal_pattern.links(i / bs[0]).rows();
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = bs[1] * off_diagonal_pattern.links(i / bs[0]).rows();
  }

  // Allocate space for matrix
  ierr = MatXAIJSetPreallocation(A, _bs, _nnz_diag.data(), _nnz_offdiag.data(),
                                 nullptr, nullptr);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatXIJSetPreallocation");

  // Set block sizes
  ierr = MatSetBlockSizes(A, bs[0], bs[1]);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetBlockSizes");

  // Create PETSc local-to-global map/index sets
  ISLocalToGlobalMapping local_to_global0;
  const std::vector _map0 = maps[0]->global_indices();
  const std::vector<PetscInt> map0(_map0.begin(), _map0.end());
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[0], map0.size(),
                                      map0.data(), PETSC_COPY_VALUES,
                                      &local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Check for common index maps
  if (maps[0] == maps[1] and bs[0] == bs[1])
  {
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global0);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
  }
  else
  {
    ISLocalToGlobalMapping local_to_global1;
    const std::vector _map1 = maps[1]->global_indices();
    const std::vector<PetscInt> map1(_map1.begin(), _map1.end());
    ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[1], map1.size(),
                                        map1.data(), PETSC_COPY_VALUES,
                                        &local_to_global1);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global1);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
    ierr = ISLocalToGlobalMappingDestroy(&local_to_global1);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  }

  // Clean up local-to-global 0
  ierr = ISLocalToGlobalMappingDestroy(&local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if A is not of type MATIS)
  // ierr = MatISSetPreallocation(A, 0, _nnz_diag.data(), 0,
  // _nnz_offdiag.data()); if (ierr != 0)
  //   petsc_error(ierr, __FILE__, "MatISSetPreallocation");

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
MatNullSpace la::create_petsc_nullspace(MPI_Comm comm,
                                        const la::VectorSpaceBasis& nullspace)
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
//-----------------------------------------------------------------------------
std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                  const std::int32_t*, const PetscScalar*)>
PETScMatrix::add_fn(Mat A)
{
  return [A, cache = std::vector<PetscInt>()](
             std::int32_t m, const std::int32_t* rows, std::int32_t n,
             const std::int32_t* cols, const PetscScalar* vals) mutable {
    PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
    cache.resize(m + n);
    std::copy(rows, rows + m, cache.begin());
    std::copy(cols, cols + n, cache.begin() + m);
    const PetscInt *_rows = cache.data(), *_cols = _rows + m;
    ierr = MatSetValuesLocal(A, m, _rows, n, _cols, vals, ADD_VALUES);
#else
    ierr = MatSetValuesLocal(A, m, rows, n, cols, vals, ADD_VALUES);
#endif

#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
    return 0;
  };
}
//-----------------------------------------------------------------------------
std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                  const std::int32_t*, const PetscScalar*)>
PETScMatrix::add_block_fn(Mat A)
{
  return [A, cache = std::vector<PetscInt>()](
             std::int32_t m, const std::int32_t* rows, std::int32_t n,
             const std::int32_t* cols, const PetscScalar* vals) mutable {
    PetscErrorCode ierr;
#ifdef PETSC_USE_64BIT_INDICES
    cache.resize(m + n);
    std::copy(rows, rows + m, cache.begin());
    std::copy(cols, cols + n, cache.begin() + m);
    const PetscInt *_rows = cache.data(), *_cols = _rows + m;
    ierr = MatSetValuesBlockedLocal(A, m, _rows, n, _cols, vals, ADD_VALUES);
#else
    ierr = MatSetValuesBlockedLocal(A, m, rows, n, cols, vals, ADD_VALUES);
#endif

#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesBlockedLocal");
#endif
    return 0;
  };
}
//-----------------------------------------------------------------------------
std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                  const std::int32_t*, const PetscScalar*)>
PETScMatrix::add_block_expand_fn(Mat A, int bs0, int bs1)
{
  if (bs0 == 1 and bs1 == 1)
    return add_fn(A);

  return [A, bs0, bs1, cache0 = std::vector<PetscInt>(),
          cache1 = std::vector<PetscInt>()](
             std::int32_t m, const std::int32_t* rows, std::int32_t n,
             const std::int32_t* cols, const PetscScalar* vals) mutable {
    PetscErrorCode ierr;
    cache0.resize(bs0 * m);
    cache1.resize(bs1 * n);
    for (std::int32_t i = 0; i < m; ++i)
      for (int k = 0; k < bs0; ++k)
        cache0[bs0 * i + k] = bs0 * rows[i] + k;
    for (std::int32_t i = 0; i < n; ++i)
      for (int k = 0; k < bs1; ++k)
        cache1[bs1 * i + k] = bs1 * cols[i] + k;

    ierr = MatSetValuesLocal(A, cache0.size(), cache0.data(), cache1.size(),
                             cache1.data(), vals, ADD_VALUES);

#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
    return 0;
  };
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern)
    : PETScOperator(create_petsc_matrix(comm, sparsity_pattern), false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(Mat A, bool inc_ref_count)
    : PETScOperator(A, inc_ref_count)
{
  // Reference count to A is incremented in base class
}
//-----------------------------------------------------------------------------
double PETScMatrix::norm(la::Norm norm_type) const
{
  assert(_matA);
  PetscErrorCode ierr;
  double value = 0.0;
  switch (norm_type)
  {
  case la::Norm::l1:
    ierr = MatNorm(_matA, NORM_1, &value);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatNorm");
    break;
  case la::Norm::linf:
    ierr = MatNorm(_matA, NORM_INFINITY, &value);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatNorm");
    break;
  case la::Norm::frobenius:
    ierr = MatNorm(_matA, NORM_FROBENIUS, &value);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatNorm");
    break;
  default:
    throw std::runtime_error("Unknown PETSc Mat norm type");
  }

  return value;
}
//-----------------------------------------------------------------------------
void PETScMatrix::apply(AssemblyType type)
{
  common::Timer timer("Apply (PETScMatrix)");

  assert(_matA);
  PetscErrorCode ierr;

  MatAssemblyType petsc_type = MAT_FINAL_ASSEMBLY;
  if (type == AssemblyType::FLUSH)
    petsc_type = MAT_FLUSH_ASSEMBLY;

  ierr = MatAssemblyBegin(_matA, petsc_type);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(_matA, petsc_type);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatAssemblyEnd");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_options_prefix(std::string options_prefix)
{
  assert(_matA);
  MatSetOptionsPrefix(_matA, options_prefix.c_str());
}
//-----------------------------------------------------------------------------
std::string PETScMatrix::get_options_prefix() const
{
  assert(_matA);
  const char* prefix = nullptr;
  MatGetOptionsPrefix(_matA, &prefix);
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_from_options()
{
  assert(_matA);
  MatSetFromOptions(_matA);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_nullspace(const la::VectorSpaceBasis& nullspace)
{
  assert(_matA);

  // Get matrix communicator
  MPI_Comm comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_matA, &comm);

  // Create PETSc nullspace
  MatNullSpace petsc_ns = create_petsc_nullspace(comm, nullspace);

  // Attach PETSc nullspace to matrix
  assert(_matA);
  PetscErrorCode ierr = MatSetNullSpace(_matA, petsc_ns);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetNullSpace");

  // Decrease reference count for nullspace by destroying
  MatNullSpaceDestroy(&petsc_ns);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_near_nullspace(const la::VectorSpaceBasis& nullspace)
{
  assert(_matA);

  // Get matrix communicator
  MPI_Comm comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_matA, &comm);

  // Create PETSc nullspace
  MatNullSpace petsc_ns = la::create_petsc_nullspace(comm, nullspace);

  // Attach near  nullspace to matrix
  assert(_matA);
  PetscErrorCode ierr = MatSetNearNullSpace(_matA, petsc_ns);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetNullSpace");

  // Decrease reference count for nullspace
  MatNullSpaceDestroy(&petsc_ns);
}
//-----------------------------------------------------------------------------
