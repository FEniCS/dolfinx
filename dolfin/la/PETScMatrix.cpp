// Copyright (C) 2004-2012 Johan Hoffman, Johan Jansson and Anders Logg
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
//
// First added:  2004
// Last changed: 2012-03-15

#ifdef HAS_PETSC

#include <iostream>
#include <sstream>
#include <iomanip>
#include <boost/assign/list_of.hpp>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include "PETScVector.h"
#include "PETScMatrix.h"
#include "GenericSparsityPattern.h"
#include "SparsityPattern.h"
#include "TensorLayout.h"
#include "PETScFactory.h"
#include "PETScCuspFactory.h"

using namespace dolfin;

class PETScMatNullSpaceDeleter
{
public:
  void operator() (MatNullSpace* ns)
  {
    if (*ns)
      MatNullSpaceDestroy(ns);
    delete ns;
  }
};

const std::map<std::string, NormType> PETScMatrix::norm_types
  = boost::assign::map_list_of("l1",        NORM_1)
                              ("linf",      NORM_INFINITY)
                              ("frobenius", NORM_FROBENIUS);

//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(bool use_gpu) : _use_gpu(use_gpu)
{
#ifndef HAS_PETSC_CUSP
  if (use_gpu)
  {
    dolfin_error("PETScMatrix.cpp",
                 "create GPU matrix",
                 "PETSc not compiled with Cusp support");
  }
#endif

  // Do nothing else
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(boost::shared_ptr<Mat> A, bool use_gpu) :
  PETScBaseMatrix(A), _use_gpu(use_gpu)
{
#ifndef HAS_PETSC_CUSP
  if (use_gpu)
  {
    dolfin_error("PETScMatrix.cpp",
                 "create GPU matrix",
                 "PETSc not compiled with Cusp support");
  }
#endif
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const PETScMatrix& A): _use_gpu(false)
{
  *this = A;
}
//-----------------------------------------------------------------------------
PETScMatrix::~PETScMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
boost::shared_ptr<GenericMatrix> PETScMatrix::copy() const
{
  boost::shared_ptr<GenericMatrix> B;
  if (!A)
    B.reset(new PETScMatrix());
  else
  {
    // Create copy of PETSc matrix
    boost::shared_ptr<Mat> _Acopy(new Mat, PETScMatrixDeleter());
    MatDuplicate(*A, MAT_COPY_VALUES, _Acopy.get());

    // Create PETScMatrix
    B.reset(new PETScMatrix(_Acopy));
  }

  return B;
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(const TensorLayout& tensor_layout)
{
  // Get global dimensions and local range
  dolfin_assert(tensor_layout.rank() == 2);
  const uint M = tensor_layout.size(0);
  const uint N = tensor_layout.size(1);
  const std::pair<uint, uint> row_range = tensor_layout.local_range(0);
  const std::pair<uint, uint> col_range = tensor_layout.local_range(1);
  const uint m = row_range.second - row_range.first;
  const uint n = col_range.second - col_range.first;
  dolfin_assert(M > 0 && N > 0 && m > 0 && n > 0);

  // Get sparsity payttern
  dolfin_assert(tensor_layout.sparsity_pattern());
  const GenericSparsityPattern& sparsity_pattern = *tensor_layout.sparsity_pattern();

  // Create matrix (any old matrix is destroyed automatically)
  if (A && !A.unique())
  {
    dolfin_error("PETScMatrix.cpp",
                 "initialize PETSc matrix",
                 "More than one object points to the underlying PETSc object");
  }
  A.reset(new Mat, PETScMatrixDeleter());

  // Initialize matrix
  if (row_range.first == 0 && row_range.second == M)
  {
    // Get number of nonzeros for each row from sparsity pattern
    dolfin_assert(tensor_layout.sparsity_pattern());
    std::vector<uint> num_nonzeros(M);
    sparsity_pattern.num_nonzeros_diagonal(num_nonzeros);

    // Create matrix
    MatCreate(PETSC_COMM_SELF, A.get());

    // Set size
    MatSetSizes(*A, M, N, M, N);

    // Set matrix type according to chosen architecture
    if (!_use_gpu)
      MatSetType(*A, MATSEQAIJ);
#ifdef HAS_PETSC_CUSP
    else
      MatSetType(*A, MATSEQAIJCUSP);
#endif

    // FIXME: Change to MatSeqAIJSetPreallicationCSR for improved performance?
    // Allocate space (using data from sparsity pattern)
    MatSeqAIJSetPreallocation(*A, PETSC_NULL, reinterpret_cast<int*>(&num_nonzeros[0]));

    // Set column indices
    /*
    const std::vector<std::vector<uint> > _column_indices
        = sparsity_pattern.diagonal_pattern(SparsityPattern::sorted);
    std::vector<int> column_indices;
    column_indices.reserve(sparsity_pattern.num_nonzeros());
    for (uint i = 0; i < _column_indices.size(); ++i)
    {
      //cout << "Row: " << i << endl;
      //for (uint j = 0; j < _column_indices[i].size(); ++j)
      //  cout << "  Col: " << _column_indices[i][j] << endl;
      column_indices.insert(column_indices.end(), _column_indices[i].begin(), _column_indices[i].end());
    }
    MatSeqAIJSetColumnIndices(*A, &column_indices[0]);
    */
  }
  else
  {
    if (_use_gpu)
    {
      not_working_in_parallel("Due to limitations in PETSc, "
          "distributed PETSc Cusp matrices");
    }

    // FIXME: Try using MatStashSetInitialSize to optimise performance

    //info("Initializing parallel PETSc matrix (MPIAIJ) of size %d x %d.", M, N);
    //info("Local range is [%d, %d] x [%d, %d].",
    //     row_range.first, row_range.second, col_range.first, col_range.second);

    // Get number of nonzeros for each row from sparsity pattern
    std::vector<uint> num_nonzeros_diagonal;
    std::vector<uint> num_nonzeros_off_diagonal;
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
  }

  // Set some options

  // Do not allow more entries than have been pre-allocated
  MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);

  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 1
  MatSetOption(*A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  #else
  MatSetOption(*A, MAT_KEEP_ZEROED_ROWS, PETSC_TRUE);
  #endif

  MatSetFromOptions(*A);

  #if PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>2
  MatSetUp(*A.get());
  #endif
}
//-----------------------------------------------------------------------------
void PETScMatrix::get(double* block, uint m, const uint* rows,
                                     uint n, const uint* cols) const
{
  // Get matrix entries (must be on this process)
  dolfin_assert(A);
  MatGetValues(*A,
               static_cast<int>(m), reinterpret_cast<const int*>(rows),
               static_cast<int>(n), reinterpret_cast<const int*>(cols),
               block);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const double* block, uint m, const uint* rows,
                                           uint n, const uint* cols)
{
  dolfin_assert(A);
  MatSetValues(*A,
               static_cast<int>(m), reinterpret_cast<const int*>(rows),
               static_cast<int>(n), reinterpret_cast<const int*>(cols),
               block, INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScMatrix::add(const double* block, uint m, const uint* rows,
                                           uint n, const uint* cols)
{
  dolfin_assert(A);
  MatSetValues(*A,
               static_cast<int>(m), reinterpret_cast<const int*>(rows),
               static_cast<int>(n), reinterpret_cast<const int*>(cols),
               block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScMatrix::axpy(double a, const GenericMatrix& A,
                       bool same_nonzero_pattern)
{
  const PETScMatrix* AA = &A.down_cast<PETScMatrix>();
  dolfin_assert(this->A);
  dolfin_assert(AA->mat());
  if (same_nonzero_pattern)
    MatAXPY(*(this->A), a, *AA->mat(), SAME_NONZERO_PATTERN);
  else
    MatAXPY(*(this->A), a, *AA->mat(), DIFFERENT_NONZERO_PATTERN);
}
//-----------------------------------------------------------------------------
void PETScMatrix::getrow(uint row, std::vector<uint>& columns,
                         std::vector<double>& values) const
{
  dolfin_assert(A);

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
void PETScMatrix::setrow(uint row, const std::vector<uint>& columns,
                         const std::vector<double>& values)
{
  dolfin_assert(A);

  // Check size of arrays
  if (columns.size() != values.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "set row of values for PETSc matrix",
                 "Number of columns and values don't match");
  }

  // Handle case n = 0
  const uint n = columns.size();
  if (n == 0)
    return;

  // Set values
  set(&values[0], 1, &row, n, &columns[0]);
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero(uint m, const uint* rows)
{
  dolfin_assert(A);

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
void PETScMatrix::ident(uint m, const uint* rows)
{
  dolfin_assert(A);

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
void PETScMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(A);

  const PETScVector& xx = x.down_cast<PETScVector>();
  PETScVector& yy = y.down_cast<PETScVector>();

  if (size(1) != xx.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute matrix-vector product with PETSc matrix",
                 "Non-matching dimensions for matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.size() == 0)
    resize(yy, 0);

  if (size(0) != yy.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute matrix-vector product with PETSc matrix",
                 "Vector for matrix-vector result has wrong size");
  }

  MatMult(*A, *xx.vec(), *yy.vec());
}
//-----------------------------------------------------------------------------
void PETScMatrix::transpmult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(A);

  const PETScVector& xx = x.down_cast<PETScVector>();
  PETScVector& yy = y.down_cast<PETScVector>();

  if (size(0) != xx.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute transpose matrix-vector product with PETSc matrix",
                 "Non-matching dimensions for transpose matrix-vector product");
  }

  // Resize RHS if empty
  if (yy.size() == 0)
    resize(yy, 1);

  if (size(1) != yy.size())
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute transpose matrix-vector product with PETSc matrix",
                 "Vector for transpose matrix-vector result has wrong size");
  }

  MatMultTranspose(*A, *xx.vec(), *yy.vec());
}
//-----------------------------------------------------------------------------
void PETScMatrix::set_near_nullspace(std::vector<const GenericVector*> nullspace)
{
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR < 3
  dolfin_error("PETScMatrix.cpp",
               "set approximate null space for PETSc matrix",
               "This is supported by PETSc version > 3.2");
  #else
  warning("PETScMatrix::set_near_nullspace is experimental and is likely to be re-named or moved in the future.");

  dolfin_assert(nullspace.size() > 0);

  // Copy vectors
  _nullspace.clear();
  for (uint i = 0; i < nullspace.size(); ++i)
  {
    dolfin_assert(nullspace[i]);
    const PETScVector& x = nullspace[i]->down_cast<PETScVector>();

    // Copy vector
    _nullspace.push_back(x);
  }

  // Get pointers to underlying PETSc objects
  std::vector<Vec> petsc_vec(nullspace.size());
  for (uint i = 0; i < nullspace.size(); ++i)
    petsc_vec[i] = *(_nullspace[i].vec().get());

  // Create null space
  petsc_nullspace.reset(new MatNullSpace, PETScMatNullSpaceDeleter());
  MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, nullspace.size(),
                     &petsc_vec[0], petsc_nullspace.get());

  //MatSetNullSpace(*(this->A), petsc_nullspace);

  // Set null space that is used by some preconditioners
  MatSetNearNullSpace(*(this->A), *petsc_nullspace);
  #endif
}
//-----------------------------------------------------------------------------
double PETScMatrix::norm(std::string norm_type) const
{
  dolfin_assert(A);

  // Check that norm is known
  if (norm_types.count(norm_type) == 0)
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute norm of PETSc matrix",
                 "Unknown norm type (\"%s\")", norm_type.c_str());
  }

  double value = 0.0;
  MatNorm(*A, norm_types.find(norm_type)->second, &value);
  return value;
}
//-----------------------------------------------------------------------------
void PETScMatrix::apply(std::string mode)
{
  Timer("Apply (matrix)");

  dolfin_assert(A);
  if (mode == "add")
  {
    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
  }
  else if (mode == "insert")
  {
    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
  }
  else if (mode == "flush")
  {
    MatAssemblyBegin(*A, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(*A, MAT_FLUSH_ASSEMBLY);
  }
  else
  {
    dolfin_error("PETScMatrix.cpp",
                 "apply changes to PETSc matrix",
                 "Unknown apply mode \"%s\"", mode.c_str());
  }
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero()
{
  dolfin_assert(A);
  MatZeroEntries(*A);
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator*= (double a)
{
  dolfin_assert(A);
  MatScale(*A, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator/= (double a)
{
  dolfin_assert(A);
  MatScale(*A, 1.0/a);
  return *this;
}
//-----------------------------------------------------------------------------
const GenericMatrix& PETScMatrix::operator= (const GenericMatrix& A)
{
  *this = A.down_cast<PETScMatrix>();
  return *this;
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator= (const PETScMatrix& A)
{
  if (!A.mat())
    this->A.reset();
  else if (this != &A) // Check for self-assignment
  {
    if (this->A && !this->A.unique())
    {
      dolfin_error("PETScMatrix.cpp",
                   "assign to PETSc matrix",
                   "More than one object points to the underlying PETSc object");
    }
    this->A.reset(new Mat, PETScMatrixDeleter());

    // Duplicate with the same pattern as A.A
    MatDuplicate(*A.mat(), MAT_COPY_VALUES, this->A.get());
  }
  return *this;
}
//-----------------------------------------------------------------------------
void PETScMatrix::binary_dump(std::string file_name) const
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
std::string PETScMatrix::str(bool verbose) const
{
  if (!A)
    return "<Uninitialized PETScMatrix>";

  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScMatrix not implemented, calling PETSc MatView directly.");

    // FIXME: Maybe this could be an option?
    dolfin_assert(A);
    if (MPI::num_processes() > 1)
      MatView(*A, PETSC_VIEWER_STDOUT_WORLD);
    else
      MatView(*A, PETSC_VIEWER_STDOUT_SELF);
  }
  else
    s << "<PETScMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& PETScMatrix::factory() const
{
  if (!_use_gpu)
    return PETScFactory::instance();
#ifdef HAS_PETSC_CUSP
  else
    return PETScCuspFactory::instance();
#endif

  // Return something to keep the compiler happy. Code will never be reached.
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------

#endif
