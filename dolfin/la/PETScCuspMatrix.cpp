// Copyright (C) 2011 Fredrik Valdmanis
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
// First added:  2011-09-13
// Last changed: 2011-09-13

//FIXME: FREDRIK: Should we test for both HAS_PETSC and PETSC_HAVE_CUSP?
#ifdef PETSC_HAVE_CUSP

#include <iostream>
#include <sstream>
#include <iomanip>
#include <boost/assign/list_of.hpp>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include "PETScCuspVector.h"
#include "PETScCuspMatrix.h"
#include "GenericSparsityPattern.h"
#include "SparsityPattern.h"
#include "PETScCuspFactory.h"

using namespace dolfin;

const std::map<std::string, NormType> PETScCuspMatrix::norm_types
  = boost::assign::map_list_of("l1",        NORM_1)
                              ("linf",      NORM_INFINITY)
                              ("frobenius", NORM_FROBENIUS);

//-----------------------------------------------------------------------------
PETScCuspMatrix::PETScCuspMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScCuspMatrix::PETScCuspMatrix(boost::shared_ptr<Mat> A) : PETScBaseMatrix(A)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScCuspMatrix::PETScCuspMatrix(const PETScCuspMatrix& A)
{
  *this = A;
}
//-----------------------------------------------------------------------------
PETScCuspMatrix::~PETScCuspMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool PETScCuspMatrix::distributed() const
{
  assert(A);

  // Get type
  const MatType petsc_type;
  MatGetType(*A, &petsc_type);

  // Return type
  bool _distributed = false;
  if (strncmp(petsc_type, "seq", 3) != 0)
    _distributed = true;

  return _distributed;
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::resize(uint M, uint N)
{
  // FIXME: Remove this function or use init() function somehow to
  // FIXME: avoid duplication of code

  if (A && size(0) == N && size(1) == N)
    return;

  // Create matrix (any old matrix is destroyed automatically)
  if (A && !A.unique())
    error("Cannot resize PETScCuspMatrix. More than one object points to the underlying PETSc object.");
  A.reset(new Mat, PETScCuspMatrixDeleter());

  // FIXME: maybe 50 should be a parameter?
  // FIXME: it should definitely be a parameter

  // Create a sparse matrix in compressed row format
  if (dolfin::MPI::num_processes() > 1)
  {
    // Create PETSc parallel matrix with a guess for number of diagonal (50 in this case)
    // and number of off-diagonal non-zeroes (50 in this case).
    // Note that guessing too high leads to excessive memory usage.
    // In order to not waste any memory one would need to specify d_nnz and o_nnz.
    MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N,
                    50, PETSC_NULL, 50, PETSC_NULL, A.get());
  }
  else
  {
    // Create PETSc sequential matrix with a guess for number of non-zeroes (50 in thise case)
    MatCreateSeqAIJ(PETSC_COMM_SELF, M, N, 50, PETSC_NULL, A.get());
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 1
    MatSetOption(*A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
    #else
    MatSetOption(*A, MAT_KEEP_ZEROED_ROWS, PETSC_TRUE);
    #endif
    MatSetFromOptions(*A);
  }
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::init(const GenericSparsityPattern& sparsity_pattern)
{
  // Get global dimensions and local range
  assert(sparsity_pattern.rank() == 2);
  const uint M = sparsity_pattern.size(0);
  const uint N = sparsity_pattern.size(1);
  const std::pair<uint, uint> row_range = sparsity_pattern.local_range(0);
  const std::pair<uint, uint> col_range = sparsity_pattern.local_range(1);
  const uint m = row_range.second - row_range.first;
  const uint n = col_range.second - col_range.first;
  assert(M > 0 && N > 0 && m > 0 && n > 0);

  // Create matrix (any old matrix is destroyed automatically)
  if (A && !A.unique())
    error("Cannot initialise PETScCuspMatrix. More than one object points to the underlying PETSc object.");
  A.reset(new Mat, PETScCuspMatrixDeleter());

  // Initialize matrix
  if (row_range.first == 0 && row_range.second == M)
  {
    // Get number of nonzeros for each row from sparsity pattern
    std::vector<uint> num_nonzeros(M);
    sparsity_pattern.num_nonzeros_diagonal(num_nonzeros);

    // Create matrix
    MatCreate(PETSC_COMM_SELF, A.get());

    // Set size
    MatSetSizes(*A, M, N, M, N);

    // Set matrix type
    MatSetType(*A, MATSEQAIJ);

    // Allocate space (using data from sparsity pattern)
    MatSeqAIJSetPreallocation(*A, PETSC_NULL, reinterpret_cast<int*>(&num_nonzeros[0]));

    /*
    // Set column indices
    std::vector<std::vector<uint> > _column_indices = sparsity_pattern.diagonal_pattern(SparsityPattern::unsorted);
    std::vector<int> column_indices(sparsity_pattern.num_nonzeros());
    uint k = 0;
    for (uint i = 0; i < _column_indices.size(); ++i)
    {
      for (uint j = 0; j < _column_indices[i].size(); ++j)
        column_indices[k++] = _column_indices[i][j];
    }
    MatSeqAIJSetColumnIndices(*A, reinterpret_cast<int*>(&column_indices[0]));
    */

    // Do not allow new nonzero entries
    //MatSetOption(*A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);

    // Set some options
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 1
    MatSetOption(*A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
    #else
    MatSetOption(*A, MAT_KEEP_ZEROED_ROWS, PETSC_TRUE);
    #endif

    MatSetFromOptions(*A);
  }
  else
  {
    // FIXME: Try using MatStashSetInitialSize to optimise performance

    //info("Initializing parallel PETSc matrix (MPIAIJ) of size %d x %d.", M, N);
    //info("Local range is [%d, %d] x [%d, %d].",
    //     row_range.first, row_range.second, col_range.first, col_range.second);

    // Get number of nonzeros for each row from sparsity pattern
    std::vector<uint> num_nonzeros_diagonal(m);
    std::vector<uint> num_nonzeros_off_diagonal(n);
    sparsity_pattern.num_nonzeros_diagonal(num_nonzeros_diagonal);
    sparsity_pattern.num_nonzeros_off_diagonal(num_nonzeros_off_diagonal);

    // Create matrix
    MatCreate(PETSC_COMM_WORLD, A.get());

    // Set size
    MatSetSizes(*A, m, n, M, N);

    // Set matrix type
    MatSetType(*A, MATMPIAIJ);

    // Allocate space (using data from sparsity pattern)
    MatMPIAIJSetPreallocation(*A,
           PETSC_NULL, reinterpret_cast<int*>(&num_nonzeros_diagonal[0]),
           PETSC_NULL, reinterpret_cast<int*>(&num_nonzeros_off_diagonal[0]));

    // Set some options
    #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 1
    MatSetOption(*A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
    #else
    MatSetOption(*A, MAT_KEEP_ZEROED_ROWS, PETSC_TRUE);
    #endif

    MatSetFromOptions(*A);
  }
}
//-----------------------------------------------------------------------------
PETScCuspMatrix* PETScCuspMatrix::copy() const
{
  if (!A)
    return new PETScCuspMatrix();
  else
  {
    // Create copy of PETSc matrix
    boost::shared_ptr<Mat> _Acopy(new Mat, PETScCuspMatrixDeleter());
    MatDuplicate(*A, MAT_COPY_VALUES, _Acopy.get());

    // Create PETScCuspMatrix
    return new PETScCuspMatrix(_Acopy);
  }
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::get(double* block, uint m, const uint* rows,
                                     uint n, const uint* cols) const
{
  assert(A);

  // Get matrix entries (must be on this process)
  MatGetValues(*A,
               static_cast<int>(m), reinterpret_cast<const int*>(rows),
               static_cast<int>(n), reinterpret_cast<const int*>(cols),
               block);
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::set(const double* block, uint m, const uint* rows,
                                           uint n, const uint* cols)
{
  assert(A);
  MatSetValues(*A,
               static_cast<int>(m), reinterpret_cast<const int*>(rows),
               static_cast<int>(n), reinterpret_cast<const int*>(cols),
               block, INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::add(const double* block, uint m, const uint* rows,
                                           uint n, const uint* cols)
{
  assert(A);
  MatSetValues(*A,
               static_cast<int>(m), reinterpret_cast<const int*>(rows),
               static_cast<int>(n), reinterpret_cast<const int*>(cols),
               block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::axpy(double a, const GenericMatrix& A,
                       bool same_nonzero_pattern)
{
  const PETScCuspMatrix* AA = &A.down_cast<PETScCuspMatrix>();
  assert(this->A);
  assert(AA->mat());
  if (same_nonzero_pattern)
    MatAXPY(*(this->A), a, *AA->mat(), SAME_NONZERO_PATTERN);
  else
    MatAXPY(*(this->A), a, *AA->mat(), DIFFERENT_NONZERO_PATTERN);
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::getrow(uint row, std::vector<uint>& columns,
                         std::vector<double>& values) const
{
  assert(A);

  const int *cols = 0;
  const double *vals = 0;
  int ncols = 0;
  MatGetRow(*A, row, &ncols, &cols, &vals);

  // Assign values to std::vectors
  columns.assign(cols, cols + ncols);
  values.assign(vals, vals + ncols);

  MatRestoreRow(*A, row, &ncols, &cols, &vals);
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::setrow(uint row, const std::vector<uint>& columns,
                         const std::vector<double>& values)
{
  assert(A);

  // Check size of arrays
  if (columns.size() != values.size())
    error("Number of columns and values don't match for setrow() operation.");

  // Handle case n = 0
  const uint n = columns.size();
  if (n == 0)
    return;

  // Set values
  set(&values[0], 1, &row, n, &columns[0]);
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::zero(uint m, const uint* rows)
{
  assert(A);

  IS is = 0;
  PetscScalar null = 0.0;
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 1
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m),
                  reinterpret_cast<const int*>(rows),
                  PETSC_COPY_VALUES, &is);
  MatZeroRowsIS(*A, is, null, NULL, NULL);
  #else
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m),
                  reinterpret_cast<const int*>(rows), &is);
  MatZeroRowsIS(*A, is, null);
  #endif

  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 1
  ISDestroy(&is);
  #else
  ISDestroy(is);
  #endif
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::ident(uint m, const uint* rows)
{
  assert(A);

  IS is = 0;
  PetscScalar one = 1.0;
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 1
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m),
                  reinterpret_cast<const int*>(rows),
                  PETSC_COPY_VALUES, &is);
  MatZeroRowsIS(*A, is, one, NULL, NULL);
  #else
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m),
                  reinterpret_cast<const int*>(rows), &is);
  MatZeroRowsIS(*A, is, one);
  #endif

  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 1
  ISDestroy(&is);
  #else
  ISDestroy(is);
  #endif
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  assert(A);

  const PETScVector& xx = x.down_cast<PETScVector>();
  PETScVector& yy = y.down_cast<PETScVector>();

  if (size(1) != xx.size())
    error("Matrix and vector dimensions don't match for matrix-vector product.");

  resize(yy, 0);
  MatMult(*A, *xx.vec(), *yy.vec());
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::transpmult(const GenericVector& x, GenericVector& y) const
{
  assert(A);

  const PETScVector& xx = x.down_cast<PETScVector>();
  PETScVector& yy = y.down_cast<PETScVector>();

  if (size(0) != xx.size())
    error("Matrix and vector dimensions don't match for matrix-vector product.");

  resize(yy, 1);
  MatMultTranspose(*A, *xx.vec(), *yy.vec());
}
//-----------------------------------------------------------------------------
double PETScCuspMatrix::norm(std::string norm_type) const
{
  assert(A);

  // Check that norm is known
  if( norm_types.count(norm_type) == 0)
    error("Unknown PETSc matrix norm type.");

  double value = 0.0;
  MatNorm(*A, norm_types.find(norm_type)->second, &value);
  return value;
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::apply(std::string mode)
{
  assert(A);
  MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::zero()
{
  assert(A);
  MatZeroEntries(*A);
}
//-----------------------------------------------------------------------------
const PETScCuspMatrix& PETScCuspMatrix::operator*= (double a)
{
  assert(A);
  MatScale(*A, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScCuspMatrix& PETScCuspMatrix::operator/= (double a)
{
  assert(A);
  MatScale(*A, 1.0 / a);
  return *this;
}
//-----------------------------------------------------------------------------
const GenericMatrix& PETScCuspMatrix::operator= (const GenericMatrix& A)
{
  *this = A.down_cast<PETScCuspMatrix>();
  return *this;
}
//-----------------------------------------------------------------------------
const PETScCuspMatrix& PETScCuspMatrix::operator= (const PETScCuspMatrix& A)
{
  if (!A.mat())
  {
    this->A.reset();
  }
  else if (this != &A) // Check for self-assignment
  {
    if (this->A && !this->A.unique())
      error("Cannot assign PETScCuspMatrix because more than one object points to the underlying PETSc object.");
    this->A.reset(new Mat, PETScCuspMatrixDeleter());

    // Duplicate with the same pattern as A.A
    MatDuplicate(*A.mat(), MAT_COPY_VALUES, this->A.get());
  }
  return *this;
}
//-----------------------------------------------------------------------------
void PETScCuspMatrix::binary_dump(std::string file_name) const
{
  PetscViewer view_out;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_name.c_str(),
                        FILE_MODE_WRITE, &view_out);
  MatView(*(A.get()), view_out);
#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 1
  PetscViewerDestroy(view_out);
#else
  PetscViewerDestroy(&view_out);
#endif
}
//-----------------------------------------------------------------------------
std::string PETScCuspMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScCuspMatrix not implemented, calling PETSc MatView directly.");

    // FIXME: Maybe this could be an option?
    assert(A);
    if (MPI::num_processes() > 1)
      MatView(*A, PETSC_VIEWER_STDOUT_WORLD);
    else
      MatView(*A, PETSC_VIEWER_STDOUT_SELF);
  }
  else
  {
    s << "<PETScCuspMatrix of size " << size(0) << " x " << size(1) << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& PETScCuspMatrix::factory() const
{
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------

#endif
