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
  std::array index_maps{sparsity_pattern.index_map(0),
                        sparsity_pattern.index_map(1)};
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
  const std::vector _map0 = index_maps[0]->global_indices(blocked);
  const std::vector _map1 = index_maps[1]->global_indices(blocked);
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
