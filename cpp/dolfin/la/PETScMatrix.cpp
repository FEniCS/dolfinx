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
PETScMatrix::PETScMatrix(MPI_Comm comm) : PETScOperator()
{
  // Create uninitialised matrix
  PetscErrorCode ierr = MatCreate(comm, &_matA);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatCreate");
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
void PETScMatrix::init(const la::SparsityPattern& sparsity_pattern)
{
  // Throw error if already initialised
  if (!empty())
  {
    throw std::runtime_error(
        "PETScMatrix may not be initialized more than once.");
    MatDestroy(&_matA);
  }

  PetscErrorCode ierr;

  // Get common::IndexMaps
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{sparsity_pattern.index_map(0), sparsity_pattern.index_map(1)}};

  // Get block sizes
  std::array<int, 2> block_sizes
      = {{index_maps[0]->block_size(), index_maps[1]->block_size()}};

  // Get global dimensions and local range
  const std::size_t M
      = block_sizes[0] * index_maps[0]->size(common::IndexMap::MapSize::GLOBAL);
  const std::size_t N
      = block_sizes[1] * index_maps[1]->size(common::IndexMap::MapSize::GLOBAL);

  const std::array<std::int64_t, 2> row_range = index_maps[0]->local_range();
  const std::array<std::int64_t, 2> col_range = index_maps[1]->local_range();
  const std::size_t m = block_sizes[0] * (row_range[1] - row_range[0]);
  const std::size_t n = block_sizes[1] * (col_range[1] - col_range[0]);

  // Get block size
  int block_size = block_sizes[0];
  if (block_sizes[0] != block_sizes[1])
  {
    log::warning(
        "Non-matching block size in PETscMatrix::init. This code needs "
        "checking.");
    block_size = 1;
  }

  // Get number of nonzeros for each row from sparsity pattern
  EigenArrayXi32 num_nonzeros_diagonal
      = sparsity_pattern.num_nonzeros_diagonal();
  EigenArrayXi32 num_nonzeros_off_diagonal
      = sparsity_pattern.num_nonzeros_off_diagonal();

  // if (block_size == 1)
  //  std::cout << "*** mat size: " << m << ", " << n << std::endl;

  // Set matrix size
  ierr = MatSetSizes(_matA, m, n, M, N);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetSizes");

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(_matA);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetFromOptions");

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _num_nonzeros_diagonal(num_nonzeros_diagonal.size()
                                               / block_size),
      _num_nonzeros_off_diagonal(num_nonzeros_off_diagonal.size() / block_size);

  for (std::size_t i = 0; i < _num_nonzeros_diagonal.size(); ++i)
  {
    _num_nonzeros_diagonal[i]
        = dolfin_ceil_div(num_nonzeros_diagonal[block_size * i], block_size);
  }
  for (std::size_t i = 0; i < _num_nonzeros_off_diagonal.size(); ++i)
  {
    _num_nonzeros_off_diagonal[i] = dolfin_ceil_div(
        num_nonzeros_off_diagonal[block_size * i], block_size);
  }

  // Allocate space (using data from sparsity pattern)
  ierr = MatXAIJSetPreallocation(_matA, block_size,
                                 _num_nonzeros_diagonal.data(),
                                 _num_nonzeros_off_diagonal.data(), NULL, NULL);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatXIJSetPreallocation");

  // Build local-to-global arrays
  assert(block_sizes[0] % block_size == 0);
  assert(block_sizes[1] % block_size == 0);
  std::vector<PetscInt> _map0, _map1;
  _map0.resize(index_maps[0]->size(common::IndexMap::MapSize::ALL)
               * (block_sizes[0] / block_size));
  _map1.resize(index_maps[1]->size(common::IndexMap::MapSize::ALL)
               * (block_sizes[1] / block_size));

  // for (std::size_t i = 0; i < _map0.size(); ++i)
  //   _map0[i] = index_maps[0]->local_to_global(i);
  // for (std::size_t i = 0; i < _map1.size(); ++i)
  //   _map1[i] = index_maps[1]->local_to_global(i);

  // std::cout << "Prep IS (0)" << std::endl;
  MPI::barrier(MPI_COMM_WORLD);
  for (std::size_t i = 0;
       i < index_maps[0]->size(common::IndexMap::MapSize::ALL); ++i)
  {
    std::size_t bs = block_sizes[0] / block_size;
    auto index = index_maps[0]->local_to_global(i);
    for (std::size_t j = 0; j < bs; ++j)
    {
      _map0[i * bs + j] = bs * index + j;
      // if (MPI::rank(MPI_COMM_WORLD) == 1)
      // {
      //   std::cout << "l2g: " << _map0[i * bs + j] << ", "
      //             << index_maps[0]->size(common::IndexMap::MapSize::ALL) <<
      //             ", "
      //             << index_maps[0]->size(common::IndexMap::MapSize::OWNED)
      //             << ", "
      //             << index_maps[0]->size(common::IndexMap::MapSize::UNOWNED)
      //             << std::endl;
      // }
    }
  }
  MPI::barrier(MPI_COMM_WORLD);

  // std::cout << "Prep IS (1)" << std::endl;
  for (std::size_t i = 0;
       i < index_maps[1]->size(common::IndexMap::MapSize::ALL); ++i)
  {
    std::size_t bs = block_sizes[1] / block_size;
    auto index = index_maps[1]->local_to_global(i);
    for (std::size_t j = 0; j < bs; ++j)
      _map1[i * bs + j] = bs * index + j;
  }
  // std::cout << "End Prep IS" << std::endl;

  /*
  if (block_size == 1)
  {
    std::cout << "** Local-to-global maps" << std::endl;
    for (std::size_t i = 0; i < _map0.size(); ++i)
      std::cout << "   " << _map0[i] << std::endl;
    std::cout << "------------------" << std::endl;
    for (std::size_t i = 0; i < _map1.size(); ++i)
      std::cout << "   " << _map1[i] << std::endl;
  }
  */

  // FIXME: In many cases the rows and columns could shared a common
  // local-to-global map

  // Create pointers to PETSc IndexSet for local-to-global map
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;

  // Create PETSc local-to-global map/index set
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, block_size, _map0.size(),
                               _map0.data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, block_size, _map1.size(),
                               _map1.data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global1);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Set matrix local-to-global maps
  // std::cout << "***** set local-to-global on mat" << std::endl;
  MatSetLocalToGlobalMapping(_matA, petsc_local_to_global0,
                             petsc_local_to_global1);
  // std::cout << "***** end set local-to-global on mat" << std::endl;
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if _matA is not of type
  // MATIS)
  ierr = MatISSetPreallocation(_matA, 0, _num_nonzeros_diagonal.data(), 0,
                               _num_nonzeros_off_diagonal.data());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatISSetPreallocation");

  // if (MPI::rank(MPI_COMM_WORLD) == 1)
  // {
  //   std::cout << "M---------------------" << std::endl;
  //   ISLocalToGlobalMappingView(petsc_local_to_global1,
  //                              PETSC_VIEWER_STDOUT_SELF);
  //   std::cout << "M---------------------" << std::endl;
  // }

  // Clean up local-to-global maps
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Set some options on _matA object

  // Do not allow more entries than have been pre-allocated
  ierr = MatSetOption(_matA, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetOption");

  // Keep nonzero structure after calling MatZeroRows
  ierr = MatSetOption(_matA, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetOption");
}
//-----------------------------------------------------------------------------
bool PETScMatrix::empty() const
{
  auto sizes = la::PETScOperator::size();
  assert((sizes[0] < 1 and sizes[1] < 1) or (sizes[0] > 0 and sizes[1] > 0));
  return (sizes[0] < 1) and (sizes[1] < 1);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> PETScMatrix::local_range(std::size_t dim) const
{
  return PETScOperator::local_range(dim);
}
//-----------------------------------------------------------------------------
void PETScMatrix::get(PetscScalar* block, std::size_t m,
                      const dolfin::la_index_t* rows, std::size_t n,
                      const dolfin::la_index_t* cols) const
{
  // Get matrix entries (must be on this process)
  assert(_matA);
  PetscErrorCode ierr = MatGetValues(_matA, m, rows, n, cols, block);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatGetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const PetscScalar* block, std::size_t m,
                      const dolfin::la_index_t* rows, std::size_t n,
                      const dolfin::la_index_t* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValues(_matA, m, rows, n, cols, block, INSERT_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_local(const PetscScalar* block, std::size_t m,
                            const dolfin::la_index_t* rows, std::size_t n,
                            const dolfin::la_index_t* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValuesLocal(_matA, m, rows, n, cols, block, INSERT_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add(const PetscScalar* block, std::size_t m,
                      const dolfin::la_index_t* rows, std::size_t n,
                      const dolfin::la_index_t* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValues(_matA, m, rows, n, cols, block, ADD_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add_local(const PetscScalar* block, std::size_t m,
                            const dolfin::la_index_t* rows, std::size_t n,
                            const dolfin::la_index_t* cols)
{
  assert(_matA);
  PetscErrorCode ierr
      = MatSetValuesLocal(_matA, m, rows, n, cols, block, ADD_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScMatrix::axpy(PetscScalar a, const PETScMatrix& A,
                       bool same_nonzero_pattern)
{
  PetscErrorCode ierr;

  assert(_matA);
  assert(A.mat());
  if (same_nonzero_pattern)
  {
    ierr = MatAXPY(_matA, a, A.mat(), SAME_NONZERO_PATTERN);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatAXPY");
  }
  else
  {
    // NOTE: Performing MatAXPY with DIFFERENT_NONZERO_PATTERN
    // destroys the local-to-global maps. We therefore assign the map
    // from *this. This is not ideal, the overloaded operations,
    // e.g. operator()+, do not allow 'same_nonzero_pattern' to be
    // set.

    // Get local-to-global map for PETSc matrix
    ISLocalToGlobalMapping rmapping0;
    ISLocalToGlobalMapping cmapping0;
    MatGetLocalToGlobalMapping(_matA, &rmapping0, &cmapping0);

    // Increase reference count to prevent destruction
    PetscObjectReference((PetscObject)rmapping0);
    PetscObjectReference((PetscObject)cmapping0);

    ierr = MatAXPY(_matA, a, A.mat(), DIFFERENT_NONZERO_PATTERN);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatAXPY");

    // Set local-to-global map and decrease reference count to maps
    MatSetLocalToGlobalMapping(_matA, rmapping0, cmapping0);
    ISLocalToGlobalMappingDestroy(&rmapping0);
    ISLocalToGlobalMappingDestroy(&cmapping0);
  }
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero(std::size_t m, const dolfin::la_index_t* rows)
{
  assert(_matA);

  PetscErrorCode ierr;
  PetscScalar null = 0.0;
  ierr = MatZeroRows(_matA, static_cast<PetscInt>(m), rows, null, NULL, NULL);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatZeroRows");
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero_local(std::size_t m, const dolfin::la_index_t* rows,
                             PetscScalar diag)
{
  assert(_matA);
  PetscErrorCode ierr;
  std::cout << "Testing m: " << m << std::endl;
  std::cout << "Testing r: " << rows[0] << std::endl;
  ierr = MatZeroRowsLocal(_matA, m, rows, diag, NULL, NULL);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatZeroRowsLocal");
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
void PETScMatrix::get_diagonal(PETScVector& x) const
{
  assert(_matA);
  const std::array<std::int64_t, 2> size = this->size();
  if (size[1] != size[0] || size[0] != x.size())
  {
    throw std::runtime_error(
        "Matrix and vector dimensions do not match for matrix-vector set");
  }

  PetscErrorCode ierr = MatGetDiagonal(_matA, x.vec());
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatGetDiagonal");
  x.update_ghosts();
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_diagonal(const PETScVector& x)
{
  assert(_matA);
  const std::array<std::int64_t, 2> size = this->size();
  if (size[1] != size[0] || size[0] != x.size())
  {
    throw std::runtime_error(
        "Matrix and vector dimensions do not match for matrix-vector set");
  }

  PetscErrorCode ierr = MatDiagonalSet(_matA, x.vec(), INSERT_VALUES);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatDiagonalSet");
  apply(AssemblyType::FINAL);
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
MPI_Comm PETScMatrix::mpi_comm() const { return PETScOperator::mpi_comm(); }
//-----------------------------------------------------------------------------
std::size_t PETScMatrix::nnz() const
{
  MatInfo info;
  MatGetInfo(_matA, MAT_GLOBAL_SUM, &info);
  return info.nz_allocated;
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
