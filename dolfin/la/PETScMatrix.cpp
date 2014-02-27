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
// Modified by Jan Blechta 2013
//
// First added:  2004
// Last changed: 2013-09-24

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

const std::map<std::string, NormType> PETScMatrix::norm_types
  = boost::assign::map_list_of("l1",        NORM_1)
                              ("linf",      NORM_INFINITY)
                              ("frobenius", NORM_FROBENIUS);

//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(bool use_gpu) : PETScBaseMatrix(NULL),
                                         _use_gpu(use_gpu)
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
PETScMatrix::PETScMatrix(Mat A) : PETScBaseMatrix(A), _use_gpu(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const PETScMatrix& A) : PETScBaseMatrix(NULL),
                                                 _use_gpu(false)
{
  *this = A;
}
//-----------------------------------------------------------------------------
PETScMatrix::~PETScMatrix()
{
  // Do nothing.
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> PETScMatrix::copy() const
{
  std::shared_ptr<GenericMatrix> B;
  if (!_A)
    B.reset(new PETScMatrix());
  else
  {
    // Create copy of PETSc matrix
    Mat _Acopy = NULL;
    PetscErrorCode ierr = MatDuplicate(_A, MAT_COPY_VALUES, &_Acopy);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatDuplicate");

    // Create PETScMatrix
    B.reset(new PETScMatrix(_Acopy));
  }

  return B;
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(const TensorLayout& tensor_layout)
{
  PetscErrorCode ierr;

  // Get global dimensions and local range
  dolfin_assert(tensor_layout.rank() == 2);
  const std::size_t M = tensor_layout.size(0);
  const std::size_t N = tensor_layout.size(1);
  const std::pair<std::size_t, std::size_t> row_range
    = tensor_layout.local_range(0);
  const std::pair<std::size_t, std::size_t> col_range
    = tensor_layout.local_range(1);
  const std::size_t m = row_range.second - row_range.first;
  const std::size_t n = col_range.second - col_range.first;
  dolfin_assert(M > 0 && N > 0 && m > 0 && n > 0);

  // Get sparsity pattern
  dolfin_assert(tensor_layout.sparsity_pattern());
  const GenericSparsityPattern& sparsity_pattern
    = *tensor_layout.sparsity_pattern();

  if (_A)
  {
    #ifdef DOLFIN_DEPRECATION_ERROR
    error("PETScMatrix may not be initialized more than once. Remove build definiton -DDOLFIN_DEPRECATION_ERROR to change this to a warning.");
    #else
    warning("PETScMatrix may not be initialized more than once. In version > 1.4, this will become an error.");
    #endif
    MatDestroy(&_A);
  }

  // Initialize matrix
  if (row_range.first == 0 && row_range.second == M)
  {
    // Get number of nonzeros for each row from sparsity pattern
    dolfin_assert(tensor_layout.sparsity_pattern());
    std::vector<std::size_t> num_nonzeros(M);
    sparsity_pattern.num_nonzeros_diagonal(num_nonzeros);

    // Create matrix
    ierr = MatCreate(PETSC_COMM_SELF, &_A);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatCreate");

    // Set size
    ierr = MatSetSizes(_A, M, N, M, N);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetSizes");

    // Set matrix type according to chosen architecture
    if (!_use_gpu)
    {
      ierr = MatSetType(_A, MATSEQAIJ);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetType");
    }
    #ifdef HAS_PETSC_CUSP
    else
    {
      ierr = MatSetType(_A, MATSEQAIJCUSP);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetType");
    }
    #endif

    // Set block size
    if (tensor_layout.block_size > 1)
    {
     ierr =  MatSetBlockSize(_A, tensor_layout.block_size);
     if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetBlockSize");
    }

    // FIXME: Change to MatSeqAIJSetPreallicationCSR for improved performance?

    // Allocate space (using data from sparsity pattern)

    // Copy number of non-zeros to PetscInt type
    const std::vector<PetscInt> _num_nonzeros(num_nonzeros.begin(),
                                              num_nonzeros.end());
    ierr = MatSeqAIJSetPreallocation(_A, 0, _num_nonzeros.data());
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatSeqAIJSetPreallocation");

    // Set column indices
    /*
    const std::vector<std::vector<std::size_t> > _column_indices
        = sparsity_pattern.diagonal_pattern(SparsityPattern::sorted);
    std::vector<int> column_indices;
    column_indices.reserve(sparsity_pattern.num_nonzeros());
    for (std::size_t i = 0; i < _column_indices.size(); ++i)
    {
      //cout << "Row: " << i << endl;
      //for (std::size_t j = 0; j < _column_indices[i].size(); ++j)
      //  cout << "  Col: " << _column_indices[i][j] << endl;
      column_indices.insert(column_indices.end(), _column_indices[i].begin(), _column_indices[i].end());
    }
    MatSeqAIJSetColumnIndices(_A, &column_indices[0]);
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

    // Get number of nonzeros for each row from sparsity pattern
    std::vector<std::size_t> num_nonzeros_diagonal;
    std::vector<std::size_t> num_nonzeros_off_diagonal;
    sparsity_pattern.num_nonzeros_diagonal(num_nonzeros_diagonal);
    sparsity_pattern.num_nonzeros_off_diagonal(num_nonzeros_off_diagonal);

    // Create matrix
    ierr = MatCreate(PETSC_COMM_WORLD, &_A);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatCreate");

    // Set size
    ierr = MatSetSizes(_A, m, n, M, N);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetSizes");

    // Set matrix type
    ierr = MatSetType(_A, MATMPIAIJ);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetType");

    // Set block size
    if (tensor_layout.block_size > 1)
    {
      ierr = MatSetBlockSize(_A, tensor_layout.block_size);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetBlockSize");
    }
    // Allocate space (using data from sparsity pattern)
    const std::vector<PetscInt>
      _num_nonzeros_diagonal(num_nonzeros_diagonal.begin(),
                             num_nonzeros_diagonal.end());
    const std::vector<PetscInt>
      _num_nonzeros_off_diagonal(num_nonzeros_off_diagonal.begin(),
                                 num_nonzeros_off_diagonal.end());
    ierr = MatMPIAIJSetPreallocation(_A, 0, _num_nonzeros_diagonal.data(),
                                     0, _num_nonzeros_off_diagonal.data());
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatMPIAIJSetPreallocation");
  }

  // Set some options

  // Do not allow more entries than have been pre-allocated
  ierr = MatSetOption(_A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetOption");

  ierr = MatSetOption(_A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetOption");

  ierr = MatSetFromOptions(_A);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetFromOptions");

  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 2
  ierr = MatSetUp(_A);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetUp");
  #endif
}
//-----------------------------------------------------------------------------
bool PETScMatrix::empty() const
{
  return _A ? true : false;
}
//-----------------------------------------------------------------------------
void PETScMatrix::get(double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols) const
{
  // Get matrix entries (must be on this process)
  dolfin_assert(_A);
  PetscErrorCode ierr = MatGetValues(_A, m, rows, n, cols, block);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(_A);
  PetscErrorCode ierr = MatSetValues(_A, m, rows, n, cols, block,
                                    INSERT_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::add(const double* block,
                      std::size_t m, const dolfin::la_index* rows,
                      std::size_t n, const dolfin::la_index* cols)
{
  dolfin_assert(_A);
  PetscErrorCode ierr = MatSetValues(_A, m, rows, n, cols, block, ADD_VALUES);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatSetValues");
}
//-----------------------------------------------------------------------------
void PETScMatrix::axpy(double a, const GenericMatrix& A,
                       bool same_nonzero_pattern)
{
  PetscErrorCode ierr;

  const PETScMatrix* AA = &as_type<const PETScMatrix>(A);
  dolfin_assert(_A);
  dolfin_assert(AA->mat());
  if (same_nonzero_pattern)
  {
    ierr = MatAXPY(_A, a, AA->mat(), SAME_NONZERO_PATTERN);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAXPY");
  }
  else
  {
    ierr = MatAXPY(_A, a, AA->mat(), DIFFERENT_NONZERO_PATTERN);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAXPY");
  }
}
//-----------------------------------------------------------------------------
void PETScMatrix::getrow(std::size_t row, std::vector<std::size_t>& columns,
                         std::vector<double>& values) const
{
  dolfin_assert(_A);

  PetscErrorCode ierr;
  const PetscInt *cols = 0;
  const double *vals = 0;
  PetscInt ncols = 0;
  ierr = MatGetRow(_A, row, &ncols, &cols, &vals);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatGetRow");

  // Assign values to std::vectors
  columns.assign(cols, cols + ncols);
  values.assign(vals, vals + ncols);

  ierr = MatRestoreRow(_A, row, &ncols, &cols, &vals);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatRestorRow");
}
//-----------------------------------------------------------------------------
void PETScMatrix::setrow(std::size_t row,
                         const std::vector<std::size_t>& columns,
                         const std::vector<double>& values)
{
  dolfin_assert(_A);

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
  set(&values[0], 1, &_row, n, _columns.data());
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(_A);

  PetscErrorCode ierr;
  IS is = 0;
  PetscScalar null = 0.0;
  const PetscInt _m = m;
  ierr = ISCreateGeneral(PETSC_COMM_SELF, _m, rows, PETSC_COPY_VALUES, &is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  ierr = MatZeroRowsIS(_A, is, null, NULL, NULL);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRowsIS");
  ierr = ISDestroy(&is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
}
//-----------------------------------------------------------------------------
void PETScMatrix::ident(std::size_t m, const dolfin::la_index* rows)
{
  dolfin_assert(_A);

  PetscErrorCode ierr;
  IS is = 0;
  PetscScalar one = 1.0;
  const PetscInt _m = m;
  ierr = ISCreateGeneral(PETSC_COMM_SELF, _m, rows, PETSC_COPY_VALUES, &is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISCreateGeneral");
  ierr = MatZeroRowsIS(_A, is, one, NULL, NULL);
  if (ierr == PETSC_ERR_ARG_WRONGSTATE)
  {
    dolfin_error("PETScMatrix.cpp",
                 "set given rows to identity matrix",
                 "some diagonal elements not preallocated "
                 "(try assembler option keep_diagonal)");
  }
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroRowsIS");

  ierr = ISDestroy(&is);
  if (ierr != 0) petsc_error(ierr, __FILE__, "ISDestroy");
}
//-----------------------------------------------------------------------------
void PETScMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(_A);

  const PETScVector& xx = as_type<const PETScVector>(x);
  PETScVector& yy = as_type<PETScVector>(y);

  if (size(1) != xx.size())
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

  PetscErrorCode ierr = MatMult(_A, xx.vec(), yy.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatMult");
}
//-----------------------------------------------------------------------------
void PETScMatrix::transpmult(const GenericVector& x, GenericVector& y) const
{
  dolfin_assert(_A);

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

  PetscErrorCode ierr = MatMultTranspose(_A, xx.vec(), yy.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatMultTranspose");
}
//-----------------------------------------------------------------------------
double PETScMatrix::norm(std::string norm_type) const
{
  dolfin_assert(_A);

  // Check that norm is known
  if (norm_types.count(norm_type) == 0)
  {
    dolfin_error("PETScMatrix.cpp",
                 "compute norm of PETSc matrix",
                 "Unknown norm type (\"%s\")", norm_type.c_str());
  }

  double value = 0.0;
  PetscErrorCode ierr = MatNorm(_A, norm_types.find(norm_type)->second,
                                &value);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatNorm");
  return value;
}
//-----------------------------------------------------------------------------
void PETScMatrix::apply(std::string mode)
{
  Timer timer("Apply (PETScMatrix)");

  dolfin_assert(_A);
  PetscErrorCode ierr;
  if (mode == "add")
  {
    ierr = MatAssemblyBegin(_A, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyBegin");
    ierr = MatAssemblyEnd(_A, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyEnd");
  }
  else if (mode == "insert")
  {
    ierr = MatAssemblyBegin(_A, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyBegin");
    ierr = MatAssemblyEnd(_A, MAT_FINAL_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyEnd");
  }
  else if (mode == "flush")
  {
    ierr = MatAssemblyBegin(_A, MAT_FLUSH_ASSEMBLY);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatAssemblyBegin");
    ierr = MatAssemblyEnd(_A, MAT_FLUSH_ASSEMBLY);
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
  dolfin_assert(_A);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_A, &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero()
{
  dolfin_assert(_A);
  PetscErrorCode ierr = MatZeroEntries(_A);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatZeroEntries");
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator*= (double a)
{
  dolfin_assert(_A);
  PetscErrorCode ierr = MatScale(_A, a);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatScale");
  return *this;
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator/= (double a)
{
  dolfin_assert(_A);
  MatScale(_A, 1.0/a);
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
  dolfin_assert(_A);
  PetscBool symmetric = PETSC_FALSE;
  PetscErrorCode ierr = MatIsSymmetric(_A, tol, &symmetric);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatIsSymmetric");
  return symmetric == PETSC_TRUE ? true : false;
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& PETScMatrix::factory() const
{
  if (!_use_gpu)
    return PETScFactory::instance();
  #ifdef HAS_PETSC_CUSP
  else
    return PETScCuspFactory::instance();
  #endif

  // Return something to keep the compiler happy. Code will never be
  // reached.
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator= (const PETScMatrix& A)
{
  if (!A.mat())
  {
    if (_A)
    {
      #ifdef DOLFIN_DEPRECATION_ERROR
      error("PETScVector may not be initialized more than once. Remove build definiton -DDOLFIN_DEPRECATION_ERROR to change this to a warning. Error is in PETScMatrix::operator=.");
      #else
      warning("PETScVector may not be initialized more than once. In version > 1.4, this will become an error. Warning is in PETScMatrix::operator=.");
      #endif
      MatDestroy(&_A);
    }
    _A = NULL;
  }
  else if (this != &A) // Check for self-assignment
  {
    if (_A)
    {
      // Get reference count to _A
      int ref_count = 0;
      PetscObjectGetReference((PetscObject)_A, &ref_count);
      if (ref_count > 1)
      {
        dolfin_error("PETScMatrix.cpp",
                     "assign to PETSc matrix",
                     "More than one object points to the underlying PETSc object");
      }
      #ifdef DOLFIN_DEPRECATION_ERROR
      error("PETScMatrix may not be initialized more than once. Remove build definiton -DDOLFIN_DEPRECATION_ERROR to change this to a warning. Error is in PETScMatrix::operator=.");
      #else
      warning("PETScMatrix may not be initialized more than once. In version > 1.4, this will become an error. Warning is in PETScMatrix::operator=.");
      #endif
      MatDestroy(&_A);
    }

    // Duplicate with the same pattern as A.A
    PetscErrorCode ierr = MatDuplicate(A.mat(), MAT_COPY_VALUES, &_A);
    if (ierr != 0) petsc_error(ierr, __FILE__, "MatDuplicate");
  }
  return *this;
}
//-----------------------------------------------------------------------------
void PETScMatrix::binary_dump(std::string file_name) const
{
  PetscErrorCode ierr;

  PetscViewer view_out;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_name.c_str(),
                               FILE_MODE_WRITE, &view_out);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PetscViewerBinaryOpen");

  ierr = MatView(_A, view_out);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatView");

  ierr = PetscViewerDestroy(&view_out);
  if (ierr != 0) petsc_error(ierr, __FILE__, "PetscViewerDestroy");
}
//-----------------------------------------------------------------------------
std::string PETScMatrix::str(bool verbose) const
{
  if (!_A)
    return "<Uninitialized PETScMatrix>";

  std::stringstream s;
  if (verbose)
  {
    warning("Verbose output for PETScMatrix not implemented, calling PETSc MatView directly.");

    // FIXME: Maybe this could be an option?
    dolfin_assert(_A);
    PetscErrorCode ierr;
    if (MPI::size(MPI_COMM_WORLD) > 1)
    {
      ierr = MatView(_A, PETSC_VIEWER_STDOUT_WORLD);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatView");
    }
    else
    {
      ierr = MatView(_A, PETSC_VIEWER_STDOUT_SELF);
      if (ierr != 0) petsc_error(ierr, __FILE__, "MatView");
    }
  }
  else
    s << "<PETScMatrix of size " << size(0) << " x " << size(1) << ">";

  return s.str();
}
//-----------------------------------------------------------------------------

#endif
