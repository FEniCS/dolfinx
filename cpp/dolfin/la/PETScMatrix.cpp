// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and Garth
// N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PETScMatrix.h"
#include "PETScVector.h"
#include "SparsityPattern.h"
#include "VectorSpaceBasis.h"
#include "utils.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

// Ceiling division of nonnegative integers
#define dolfin_ceil_div(x, y) (x / y + int(x % y != 0))

using namespace dolfin;
using namespace dolfin::la;

//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix() : PETScOperator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern)
{
  PetscErrorCode ierr;
  ierr = MatCreate(comm, &_matA);
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
  ierr = MatSetSizes(_matA, m, n, M, N);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetSizes");

  // Get number of nonzeros for each row from sparsity pattern
  EigenArrayXi32 nnz_diag = sparsity_pattern.num_nonzeros_diagonal();
  EigenArrayXi32 nnz_offdiag = sparsity_pattern.num_nonzeros_off_diagonal();

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(_matA);
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
  ierr = MatXAIJSetPreallocation(_matA, bs, _nnz_diag.data(),
                                 _nnz_offdiag.data(), NULL, NULL);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatXIJSetPreallocation");

  // Build and set local-to-global maps
  assert(bs0 % bs == 0);
  assert(bs1 % bs == 0);
  std::vector<PetscInt> map0((m + index_maps[0]->num_ghosts()) * (bs0 / bs));
  std::vector<PetscInt> map1((n + index_maps[1]->num_ghosts()) * (bs1 / bs));

  const int row_block_size
      = index_maps[0]->size_local() + index_maps[0]->num_ghosts();
  for (int i = 0; i < row_block_size; ++i)
  {
    std::size_t factor = bs0 / bs;
    auto index = index_maps[0]->local_to_global(i);
    for (std::size_t j = 0; j < factor; ++j)
      map0[i * factor + j] = factor * index + j;
  }

  const int col_block_size
      = index_maps[1]->size_local() + index_maps[1]->num_ghosts();
  for (int i = 0; i < col_block_size; ++i)
  {
    std::size_t factor = bs1 / bs;
    auto index = index_maps[1]->local_to_global(i);
    for (std::size_t j = 0; j < factor; ++j)
      map1[i * factor + j] = factor * index + j;
  }

  // FIXME: In many cases the rows and columns could shared a common
  // local-to-global map

  // Create pointers to PETSc IndexSet for local-to-global map
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;

  // Create PETSc local-to-global map/index set
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs, map0.size(), map0.data(),
                               PETSC_COPY_VALUES, &petsc_local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs, map1.size(), map1.data(),
                               PETSC_COPY_VALUES, &petsc_local_to_global1);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Set matrix local-to-global maps
  MatSetLocalToGlobalMapping(_matA, petsc_local_to_global0,
                             petsc_local_to_global1);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if _matA is not of type MATIS)
  ierr = MatISSetPreallocation(_matA, 0, _nnz_diag.data(), 0,
                               _nnz_offdiag.data());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatISSetPreallocation");

  // Clean up local-to-global maps
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Set some options on _matA object
  ierr = MatSetOption(_matA, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetOption");
  ierr = MatSetOption(_matA, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetOption");
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(Mat A) : PETScOperator(A)
{
  // Reference count to A is incremented in base class
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const PETScMatrix& A) : PETScOperator()
{
  assert(A.mat());
  if (!A.empty())
  {
    PetscErrorCode ierr = MatDuplicate(A.mat(), MAT_COPY_VALUES, &_matA);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatDuplicate");
  }
  else
  {
    // Create uninitialised matrix
    PetscErrorCode ierr = MatCreate(A.mpi_comm(), &_matA);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatCreate");
  }
}
//-----------------------------------------------------------------------------
PETScMatrix::~PETScMatrix()
{
  // Do nothing (PETSc matrix is destroyed in base class)
}
//-----------------------------------------------------------------------------
bool PETScMatrix::empty() const { return _matA == nullptr ? true : false; }
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScMatrix::local_range(std::size_t dim) const
{
  return PETScOperator::local_range(dim);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const PetscScalar* block, std::size_t m,
                      const PetscInt* rows, std::size_t n, const PetscInt* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValues(_matA, m, rows, n, cols, block, INSERT_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_local(const PetscScalar* block, std::size_t m,
                            const PetscInt* rows, std::size_t n,
                            const PetscInt* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValuesLocal(_matA, m, rows, n, cols, block, INSERT_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add(const PetscScalar* block, std::size_t m,
                      const PetscInt* rows, std::size_t n, const PetscInt* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValues(_matA, m, rows, n, cols, block, ADD_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add_local(const PetscScalar* block, std::size_t m,
                            const PetscInt* rows, std::size_t n,
                            const PetscInt* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValuesLocal(_matA, m, rows, n, cols, block, ADD_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScMatrix::mult(const PETScVector& x, PETScVector& y) const
{
  assert(_matA);
  if (y.size() == 0)
    y = init_vector(0);

  PetscErrorCode ierr = MatMult(_matA, x.vec(), y.vec());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatMult");
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
void PETScMatrix::zero()
{
  assert(_matA);
  PetscErrorCode ierr = MatZeroEntries(_matA);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatZeroEntries");
}
//-----------------------------------------------------------------------------
void PETScMatrix::scale(PetscScalar a)
{
  assert(_matA);
  PetscErrorCode ierr = MatScale(_matA, a);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatScale");
}
//-----------------------------------------------------------------------------
bool PETScMatrix::is_symmetric(double tol) const
{
  assert(_matA);
  PetscBool symmetric = PETSC_FALSE;
  PetscErrorCode ierr = MatIsSymmetric(_matA, tol, &symmetric);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatIsSymmetric");
  return symmetric == PETSC_TRUE ? true : false;
}
//-----------------------------------------------------------------------------
bool PETScMatrix::is_hermitian(double tol) const
{
  assert(_matA);
  PetscBool hermitian = PETSC_FALSE;
  PetscErrorCode ierr = MatIsHermitian(_matA, tol, &hermitian);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatIsHermitian");
  return hermitian == PETSC_TRUE ? true : false;
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
  const char* prefix = NULL;
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
  // Create PETSc nullspace
  MatNullSpace petsc_ns = create_petsc_nullspace(nullspace);

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
  // Create PETSc nullspace
  MatNullSpace petsc_ns = create_petsc_nullspace(nullspace);

  // Attach near  nullspace to matrix
  assert(_matA);
  PetscErrorCode ierr = MatSetNearNullSpace(_matA, petsc_ns);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetNullSpace");

  // Decrease reference count for nullspace
  MatNullSpaceDestroy(&petsc_ns);
}
//-----------------------------------------------------------------------------
std::string PETScMatrix::str(bool verbose) const
{
  assert(_matA);
  if (this->empty())
    return "<Uninitialized PETScMatrix>";

  std::stringstream s;
  if (verbose)
  {
    log::warning(
        "Verbose output for PETScMatrix not implemented, calling PETSc "
        "MatView directly.");

    // FIXME: Maybe this could be an option?
    assert(_matA);
    PetscErrorCode ierr;
    if (MPI::size(mpi_comm()) > 1)
    {
      ierr = MatView(_matA, PETSC_VIEWER_STDOUT_WORLD);
      if (ierr != 0)
        petsc_error(ierr, __FILE__, "MatView");
    }
    else
    {
      ierr = MatView(_matA, PETSC_VIEWER_STDOUT_SELF);
      if (ierr != 0)
        petsc_error(ierr, __FILE__, "MatView");
    }
  }
  else
  {
    const std::array<std::int64_t, 2> size = this->size();
    s << "<PETScMatrix of size " << size[0] << " x " << size[1] << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
MatNullSpace
PETScMatrix::create_petsc_nullspace(const la::VectorSpaceBasis& nullspace) const
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
  ierr = MatNullSpaceCreate(mpi_comm(), PETSC_FALSE, _nullspace.size(),
                            _nullspace.data(), &petsc_nullspace);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatNullSpaceCreate");

  return petsc_nullspace;
}
//-----------------------------------------------------------------------------
