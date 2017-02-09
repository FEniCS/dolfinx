// Copyright (C) 2004-2012 Johan Hoffman, Johan Jansson, Anders Logg
// and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2005-2009.
// Modified by Andy R. Terrel 2005.
// Modified by Ola Skavhaug 2007-2009.
// Modified by Magnus Vikstrøm 2007-2008.
// Modified by Fredrik Valdmanis 2011-2012
// Modified by Jan Blechta 2013
// Modified by Martin Sandve Alnæs 2014

#ifdef HAS_PETSC

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include <dolfin/log/log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include "PETScFactory.h"
#include "PETScVector.h"
#include "SparsityPattern.h"
#include "TensorLayout.h"
#include "VectorSpaceBasis.h"
#include "PETScMatrix.h"


// Ceiling division of nonnegative integers
#define dolfin_ceil_div(x, y) (x/y + int(x%y != 0))


using namespace dolfin;

const std::map<std::string, NormType> PETScMatrix::norm_types
= { {"l1",        NORM_1},
    {"linf",      NORM_INFINITY},
    {"frobenius", NORM_FROBENIUS} };
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix() : PETScMatrix(MPI_COMM_WORLD)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(MPI_Comm comm) : PETScBaseMatrix()
{
  // Create uninitialised matrix
  PetscErrorCode ierr = MatCreate(comm, &_matA);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatCreate");
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(Mat A) : PETScBaseMatrix(A)
{
  // Reference count to A is incremented in base class
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const PETScMatrix& A) : PETScBaseMatrix()
{
  dolfin_assert(A.mat());
  if (!A.empty())
  {
    PetscErrorCode ierr = MatDuplicate(A.mat(), MAT_COPY_VALUES, &_matA);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatDuplicate");
  }
  else
  {
    // Create uninitialised matrix
    PetscErrorCode ierr = MatCreate(A.mpi_comm(), &_matA);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatCreate");
  }
}
//-----------------------------------------------------------------------------
PETScMatrix::~PETScMatrix()
{
  // Do nothing (PETSc matrix is destroyed in base class)
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> PETScMatrix::copy() const
{
  return std::make_shared<PETScMatrix>(*this);
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(const TensorLayout& tensor_layout)
{
  // Throw error if already initialised
  if (!empty())
  {
    dolfin_error("PETScMatrix.cpp",
                 "init PETSc matrix",
                 "PETScMatrix may not be initialized more than once.");
    MatDestroy(&_matA);
  }

  PetscErrorCode ierr;

  // Get global dimensions and local range
  dolfin_assert(tensor_layout.rank() == 2);
  const std::size_t M = tensor_layout.size(0);
  const std::size_t N = tensor_layout.size(1);
  const std::pair<std::int64_t, std::int64_t> row_range
    = tensor_layout.local_range(0);
  const std::pair<std::int64_t, std::int64_t> col_range
    = tensor_layout.local_range(1);
  const std::size_t m = row_range.second - row_range.first;
  const std::size_t n = col_range.second - col_range.first;

  // Get sparsity pattern
  auto sparsity_pattern = tensor_layout.sparsity_pattern();

  // Get block size
  int block_size = tensor_layout.index_map(0)->block_size();
  if (block_size != tensor_layout.index_map(1)->block_size())
    block_size = 1;

  // Get number of nonzeros for each row from sparsity pattern
  std::vector<std::size_t> num_nonzeros_diagonal, num_nonzeros_off_diagonal;
  sparsity_pattern->num_nonzeros_diagonal(num_nonzeros_diagonal);
  sparsity_pattern->num_nonzeros_off_diagonal(num_nonzeros_off_diagonal);

  // Set matrix size
  ierr = MatSetSizes(_matA, m, n, M, N);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetSizes");

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(_matA);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetFromOptions");

  // Build data to initialixe sparsity pattern (modify for block size)
  std::vector<PetscInt> _num_nonzeros_diagonal(num_nonzeros_diagonal.size()/block_size),
    _num_nonzeros_off_diagonal(num_nonzeros_off_diagonal.size()/block_size);

  for (std::size_t i = 0; i < _num_nonzeros_diagonal.size(); ++i)
  {
    _num_nonzeros_diagonal[i]
      = dolfin_ceil_div(num_nonzeros_diagonal[block_size*i], block_size);
  }
  for (std::size_t i = 0; i < _num_nonzeros_off_diagonal.size(); ++i)
  {
    _num_nonzeros_off_diagonal[i]
      = dolfin_ceil_div(num_nonzeros_off_diagonal[block_size*i], block_size);
  }

  // Allocate space (using data from sparsity pattern)
  ierr = MatXAIJSetPreallocation(_matA, block_size,
                                 _num_nonzeros_diagonal.data(),
                                 _num_nonzeros_off_diagonal.data(), NULL, NULL);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatXIJSetPreallocation");


  // Create pointers to PETSc IndexSet for local-to-globa map
  ISLocalToGlobalMapping petsc_local_to_global0, petsc_local_to_global1;
  dolfin_assert(tensor_layout.rank() == 2);

  std::vector<PetscInt> _map0, _map1;
  _map0.resize(tensor_layout.index_map(0)->size(IndexMap::MapSize::ALL)/block_size);
  _map1.resize(tensor_layout.index_map(1)->size(IndexMap::MapSize::ALL)/block_size);

  for (std::size_t i = 0; i < _map0.size(); ++i)
    _map0[i] = tensor_layout.index_map(0)->local_to_global(i*block_size)/block_size;
  for (std::size_t i = 0; i < _map1.size(); ++i)
    _map1[i] = tensor_layout.index_map(1)->local_to_global(i*block_size)/block_size;

  // FIXME: In many cases the rows and columns could shared a commin
  // local-to-global map

  // Create PETSc local-to-global map/index set
  ISLocalToGlobalMappingCreate(mpi_comm(), block_size, _map0.size(), _map0.data(),
                               PETSC_COPY_VALUES, &petsc_local_to_global0);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  ISLocalToGlobalMappingCreate(mpi_comm(), block_size, _map1.size(), _map1.data(),
                               PETSC_COPY_VALUES, &petsc_local_to_global1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Set matrix local-to-global maps
  MatSetLocalToGlobalMapping(_matA, petsc_local_to_global0,
                             petsc_local_to_global1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetLocalToGlobalMapping");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if _matA is not of type
  // MATIS)
  ierr = MatISSetPreallocation(_matA, 0, _num_nonzeros_diagonal.data(),
                               0, _num_nonzeros_off_diagonal.data());
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatISSetPreallocation");

  // Clean up local-to-global maps
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Set some options on _matA object

  // Do not allow more entries than have been pre-allocated
  ierr = MatSetOption(_matA, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetOption");

  // Keep nonzero structure after calling MatZeroRows
  ierr = MatSetOption(_matA, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetOption");
}
//-----------------------------------------------------------------------------
bool PETScMatrix::empty() const
{
  auto sizes = PETScBaseMatrix::size();
  dolfin_assert((sizes.first < 1 and sizes.second < 1) or (sizes.first > 0 and sizes.second > 0));
  return (sizes.first < 1) and (sizes.second < 1);
}
//-----------------------------------------------------------------------------
void PETScMatrix::get(double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols) const
{
  // Get matrix entries (must be on this process)
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatGetValues(_matA, m, rows, n, cols, block);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatSetValues(_matA, m, rows, n, cols, block,
                                    INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_local(const double* block,
                            std::size_t m, const dolfin::la_index* rows,
                            std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatSetValuesLocal(_matA, m, rows, n, cols, block,
                                          INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add(const double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatSetValues(_matA, m, rows, n, cols, block,
                                     ADD_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add_local(const double* block,
                            std::size_t m, const dolfin::la_index* rows,
                            std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatSetValuesLocal(_matA, m, rows, n, cols, block,
                                          ADD_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetValuesLocal");
}
//-----------------------------------------------------------------------------
void PETScMatrix::axpy(double a, const GenericMatrix& A,
                       bool same_nonzero_pattern)
{
  PetscErrorCode ierr;

  const PETScMatrix* AA = &as_type<const PETScMatrix>(A);
  dolfin_assert(_matA);
  dolfin_assert(AA->mat());
  if (same_nonzero_pattern)
  {
    ierr = MatAXPY(_matA, a, AA->mat(), SAME_NONZERO_PATTERN);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAXPY");
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
    PetscObjectReference((PetscObject) rmapping0);
    PetscObjectReference((PetscObject) cmapping0);

    ierr = MatAXPY(_matA, a, AA->mat(), DIFFERENT_NONZERO_PATTERN);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAXPY");

    // Set local-to-global map and decrease reference count to maps
    MatSetLocalToGlobalMapping(_matA, rmapping0, cmapping0);
    ISLocalToGlobalMappingDestroy(&rmapping0);
    ISLocalToGlobalMappingDestroy(&cmapping0);
  }
}
//-----------------------------------------------------------------------------
void PETScMatrix::getrow(std::size_t row, std::vector<std::size_t>& columns,
                         std::vector<double>& values) const
{
  dolfin_assert(_matA);

  PetscErrorCode ierr;
  const PetscInt *cols = 0;
  const double *vals = 0;
  PetscInt ncols = 0;
  ierr = MatGetRow(_matA, row, &ncols, &cols, &vals);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetRow");

  // Assign values to std::vectors
  columns.assign(cols, cols + ncols);
  values.assign(vals, vals + ncols);

  ierr = MatRestoreRow(_matA, row, &ncols, &cols, &vals);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatRestorRow");
}
//-----------------------------------------------------------------------------
void PETScMatrix::setrow(std::size_t row,
                         const std::vector<std::size_t>& columns,
                         const std::vector<double>& values)
{
  dolfin_assert(_matA);

  // Check size of arrays
  if (columns.size() != values.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "set row of values for PETSc matrix",
                 "Number of columns and values don't match");
  }

  // Handle case n = 0
  const PetscInt n = columns.size();
  if (n == 0)
    return;

  // Set values
  const PetscInt _row = row;
  const std::vector<PetscInt> _columns(columns.begin(), columns.end());
  set(values.data(), 1, &_row, n, _columns.data());
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(_matA);

  PetscErrorCode ierr;
  PetscScalar null = 0.0;
  ierr = MatZeroRows(_matA, static_cast<PetscInt>(m), rows, null, NULL, NULL);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero_local(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(_matA);

  PetscErrorCode ierr;
  PetscScalar null = 0.0;
  ierr = MatZeroRowsLocal(_matA, static_cast<PetscInt>(m), rows, null, NULL, NULL);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRowsLocal");
}

//-----------------------------------------------------------------------------
void PETScMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(_matA);

  PetscErrorCode ierr;
  PetscScalar one = 1.0;
  ierr = MatZeroRows(_matA, m, rows, one, NULL, NULL);
  if (ierr == PETSC_ERR_ARG_WRONGSTATE)
  {
    dolfin_error("PETScMatrix.cpp",
                 "set given (global) rows to identity matrix",
                 "some diagonal elements not preallocated "
                 "(try assembler option keep_diagonal)");
  }
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRows");
}
//-----------------------------------------------------------------------------
void PETScMatrix::ident_local(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(_matA);

  PetscErrorCode ierr;
  PetscScalar one = 1.0;
  ierr = MatZeroRowsLocal(_matA, static_cast<PetscInt>(m), rows, one, NULL, NULL);
  if (ierr == PETSC_ERR_ARG_WRONGSTATE)
  {
    dolfin_error("PETScMatrix.cpp",
                 "set given (local) rows to identity matrix",
                 "some diagonal elements not preallocated "
                 "(try assembler option keep_diagonal)");
  }
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRowsLocal");
}
//-----------------------------------------------------------------------------
void PETScMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(_matA);

  const PETScVector& xx = as_type<const PETScVector>(x);
  PETScVector& yy = as_type<PETScVector>(y);

  if (this->size(1) != xx.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute matrix-vector product with PETSc matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.size() == 0)
    init_vector(yy, 0);

  if (size(0) != yy.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute matrix-vector product with PETSc matrix",
                 "Vector for matrix-vector result has wrong size");
  }

  PetscErrorCode ierr = MatMult(_matA, xx.vec(), yy.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatMult");
}
//-----------------------------------------------------------------------------
void PETScMatrix::transpmult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(_matA);

  const PETScVector& xx = as_type<const PETScVector>(x);
  PETScVector& yy = as_type<PETScVector>(y);

  if (size(0) != xx.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute transpose matrix-vector product with PETSc matrix",
                 "Non-matching dimensions for transpose matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.size() == 0)
    init_vector(yy, 1);

  if (size(1) != yy.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute transpose matrix-vector product with PETSc matrix",
                 "Vector for transpose matrix-vector result has wrong size");
  }

  PetscErrorCode ierr = MatMultTranspose(_matA, xx.vec(), yy.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatMultTranspose");
}
//-----------------------------------------------------------------------------
void PETScMatrix::get_diagonal(GenericVector& x) const
{
  dolfin_assert(_matA);

  PETScVector& xx = x.down_cast<PETScVector>();
  if (size(1) != size(0) || size(0) != xx.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "get diagonal of a PETSc matrix",
                 "Matrix and vector dimensions don't match for matrix-vector set");
  }

  PetscErrorCode ierr = MatGetDiagonal(_matA, xx.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetDiagonal");
  xx.update_ghost_values();
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_diagonal(const GenericVector& x)
{
  dolfin_assert(_matA);

  const PETScVector& xx = x.down_cast<PETScVector>();
  if (size(1) != size(0) || size(0) != xx.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "set diagonal of a PETSc matrix",
                 "Matrix and vector dimensions don't match for matrix-vector set");
  }

  PetscErrorCode ierr = MatDiagonalSet(_matA, xx.vec(), INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatDiagonalSet");
  apply("insert");
}
//-----------------------------------------------------------------------------
double PETScMatrix::norm(std::string norm_type) const
{
  dolfin_assert(_matA);

  // Check that norm is known
  if (norm_types.count(norm_type) == 0)
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute norm of PETSc matrix",
                 "Unknown norm type (\"%s\")", norm_type.c_str());
  }

  double value = 0.0;
  PetscErrorCode ierr = MatNorm(_matA, norm_types.find(norm_type)->second,
                                &value);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatNorm");
  return value;
}
//-----------------------------------------------------------------------------
void PETScMatrix::apply(std::string mode)
{
  Timer timer("Apply (PETScMatrix)");

  dolfin_assert(_matA);
  PetscErrorCode ierr;
  if (mode == "add")
  {
    ierr = MatAssemblyBegin(_matA, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyBegin");
    ierr = MatAssemblyEnd(_matA, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyEnd");
  }
  else if (mode == "insert")
  {
    ierr = MatAssemblyBegin(_matA, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyBegin");
    ierr = MatAssemblyEnd(_matA, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyEnd");
  }
  else if (mode == "flush")
  {
    ierr = MatAssemblyBegin(_matA, MAT_FLUSH_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyBegin");
    ierr = MatAssemblyEnd(_matA, MAT_FLUSH_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyEnd");
  }
  else
  {
    dolfin_error("PETScMatrix.cpp",
                 "apply changes to PETSc matrix",
                 "Unknown apply mode \"%s\"", mode.c_str());
  }
}
//-----------------------------------------------------------------------------
MPI_Comm PETScMatrix::mpi_comm() const
{
  return PETScBaseMatrix::mpi_comm();
}
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
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatZeroEntries(_matA);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroEntries");
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator*= (double a)
{
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatScale(_matA, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatScale");
  return *this;
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator/= (double a)
{
  dolfin_assert(_matA);
  MatScale(_matA, 1.0/a);
  return *this;
}
//-----------------------------------------------------------------------------
const GenericMatrix& PETScMatrix::operator= (const GenericMatrix& A)
{
  *this = as_type<const PETScMatrix>(A);
  return *this;
}
//-----------------------------------------------------------------------------
bool PETScMatrix::is_symmetric(double tol) const
{
  dolfin_assert(_matA);
  PetscBool symmetric = PETSC_FALSE;
  PetscErrorCode ierr = MatIsSymmetric(_matA, tol, &symmetric);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatIsSymmetric");
  return symmetric == PETSC_TRUE ? true : false;
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& PETScMatrix::factory() const
{
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_options_prefix(std::string options_prefix)
{
  dolfin_assert(_matA);
  MatSetOptionsPrefix(_matA, options_prefix.c_str());
}
//-----------------------------------------------------------------------------
std::string PETScMatrix::get_options_prefix() const
{
  dolfin_assert(_matA);
  const char* prefix = NULL;
  MatGetOptionsPrefix(_matA, &prefix);
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_from_options()
{
  dolfin_assert(_matA);
  MatSetFromOptions(_matA);
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator= (const PETScMatrix& A)
{
  if (!A.mat())
  {
    if (_matA)
    {
      dolfin_error("PETScMatrix.cpp",
                   "assign to PETSc matrix",
                   "PETScMatrix may not be initialized more than once.");
      MatDestroy(&_matA);
    }
    _matA = NULL;
  }
  else if (this != &A) // Check for self-assignment
  {
    if (_matA)
    {
      // Get reference count to _matA
      PetscInt ref_count = 0;
      PetscObjectGetReference((PetscObject)_matA, &ref_count);
      if (ref_count > 1)
      {
        dolfin_error("PETScMatrix.cpp",
                     "assign to PETSc matrix",
                     "More than one object points to the underlying PETSc object");
      }
      dolfin_error("PETScMatrix.cpp",
                   "assign to PETSc matrix",
                   "PETScMatrix may not be initialized more than once.");
      MatDestroy(&_matA);
    }

    // Duplicate with the same pattern as A.A
    PetscErrorCode ierr = MatDuplicate(A.mat(), MAT_COPY_VALUES, &_matA);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatDuplicate");
  }
  return *this;
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_nullspace(const VectorSpaceBasis& nullspace)
{
  // Build PETSc nullspace
  MatNullSpace petsc_ns = create_petsc_nullspace(nullspace);

  // Attach PETSc nullspace to matrix
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatSetNullSpace(_matA, petsc_ns);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetNullSpace");

  // Decrease reference count for nullspace by destroying
  MatNullSpaceDestroy(&petsc_ns);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_near_nullspace(const VectorSpaceBasis& nullspace)
{
  // Create PETSc nullspace
  MatNullSpace petsc_ns = create_petsc_nullspace(nullspace);

  // Attach near  nullspace to matrix
  dolfin_assert(_matA);
  PetscErrorCode ierr = MatSetNearNullSpace(_matA, petsc_ns);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetNullSpace");

  // Decrease reference count for nullspace
  MatNullSpaceDestroy(&petsc_ns);
}
//-----------------------------------------------------------------------------
void PETScMatrix::binary_dump(std::string file_name) const
{
  PetscErrorCode ierr;

  PetscViewer view_out;
  ierr = PetscViewerBinaryOpen(mpi_comm(), file_name.c_str(),
                               FILE_MODE_WRITE, &view_out);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PetscViewerBinaryOpen");

  ierr = MatView(_matA, view_out);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatView");

  ierr = PetscViewerDestroy(&view_out);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PetscViewerDestroy");
}
//-----------------------------------------------------------------------------
std::string PETScMatrix::str(bool verbose) const
{
  dolfin_assert(_matA);
  if (this->empty())
    return "<Uninitialized PETScMatrix>";

  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for PETScMatrix not implemented, calling PETSc MatView directly.");

    // FIXME: Maybe this could be an option?
    dolfin_assert(_matA);
    PetscErrorCode ierr;
    if (MPI::size(mpi_comm()) > 1)
    {
      ierr = MatView(_matA, PETSC_VIEWER_STDOUT_WORLD);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatView");
    }
    else
    {
      ierr = MatView(_matA, PETSC_VIEWER_STDOUT_SELF);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatView");
    }
  }
  else
    s << "<PETScMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
MatNullSpace PETScMatrix::create_petsc_nullspace(const VectorSpaceBasis& nullspace) const
{
  PetscErrorCode ierr;

  // Copy vectors in vector space object
  std::vector<PETScVector> _nullspace;
  for (std::size_t i = 0; i < nullspace.dim(); ++i)
  {
    dolfin_assert(nullspace[i]);
    const PETScVector& x = nullspace[i]->down_cast<PETScVector>();

    // Copy vector
    _nullspace.push_back(x);
  }

  // Get pointers to underlying PETSc objects
  std::vector<Vec> petsc_vecs;
  for (auto& basis_vector : _nullspace)
    petsc_vecs.push_back(basis_vector.vec());

  // Create PETSC nullspace
  MatNullSpace petsc_nullspace = NULL;
  ierr = MatNullSpaceCreate(mpi_comm(), PETSC_FALSE, petsc_vecs.size(),
                            petsc_vecs.data(), &petsc_nullspace);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatNullSpaceCreate");

  return petsc_nullspace;
}
//-----------------------------------------------------------------------------


#endif
