// Copyright (C) 2004-2008 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2009.
// Modified by Andy R. Terrel 2005.
// Modified by Ola Skavhaug 2007-2009.
// Modified by Magnus Vikstrøm 2007-2008.
//
// First added:  2004
// Last changed: 2009-09-08

#ifdef HAS_PETSC

#include <iostream>
#include <sstream>
#include <iomanip>
#include <boost/assign/list_of.hpp>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/main/MPI.h>
#include "PETScVector.h"
#include "PETScMatrix.h"
#include "GenericSparsityPattern.h"
#include "SparsityPattern.h"
#include "PETScFactory.h"

namespace dolfin
{
  class PETScMatrixDeleter
  {
  public:
    void operator() (Mat* A)
    {
      if (A)
        MatDestroy(*A);
      delete A;
    }
  };
}

using namespace dolfin;

const std::map<std::string, NormType> PETScMatrix::norm_types
  = boost::assign::map_list_of("l1",        NORM_1)
                              ("lif",       NORM_INFINITY)
                              ("frobenius", NORM_FROBENIUS);

//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(boost::shared_ptr<Mat> A) : A(A)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(uint M, uint N)
{
  // Create PETSc matrix
  resize(M, N);
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const PETScMatrix& A)
{
  *this = A;
}
//-----------------------------------------------------------------------------
PETScMatrix::~PETScMatrix()
{
  // Do nothing. The custom shared_ptr deleter takes care of the cleanup.
}
//-----------------------------------------------------------------------------
void PETScMatrix::resize(uint M, uint N)
{
  // FIXME: Remove this function or use init() function somehow to
  // FIXME: avoid duplication of code

  if (A && size(0) == N && size(1) == N)
    return;

  // Create matrix (any old matrix is destroyed automatically)
  if (A && !A.unique())
    error("Cannot resize PETScMatrix. More than one object points to the underlying PETSc object.");
  A.reset(new Mat, PETScMatrixDeleter());

  // FIXME: maybe 50 should be a parameter?
  // FIXME: it should definitely be a parameter

  // Create a sparse matrix in compressed row format
  if (dolfin::MPI::num_processes() > 1)
  {
    // Create PETSc parallel matrix with a guess for number of diagonal (50 in this case)
    // and number of off-diagonal non-zeroes (50 in this case).
    // Note that guessing too high leads to excessive memory usage.
    // In order to not waste any memory one would need to specify d_nnz and o_nnz.
    MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, 50, PETSC_NULL, 50, PETSC_NULL, A.get());
  }
  else
  {
    // Create PETSc sequential matrix with a guess for number of non-zeroes (50 in thise case)
    MatCreateSeqAIJ(PETSC_COMM_SELF, M, N, 50, PETSC_NULL, A.get());
    MatSetOption(*A, MAT_KEEP_ZEROED_ROWS, PETSC_TRUE);
    MatSetFromOptions(*A);
  }
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(const GenericSparsityPattern& sparsity_pattern)
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
    error("Cannot initialise PETScMatrix. More than one object points to the underlying PETSc object.");
  A.reset(new Mat, PETScMatrixDeleter());

  // Initialize matrix
  if (row_range.first == 0 && row_range.second == M)
  {
    // Get number of nonzeros for each row from sparsity pattern
    uint* num_nonzeros = new uint[M];
    assert(num_nonzeros);
    sparsity_pattern.num_nonzeros_diagonal(num_nonzeros);

    // FIXME: Does this need to be cleaned up? Seems like a mix of
    // FIXME: of MatSeqAIJ and MatSetFromOptions?

    // FIXME: It can be cleaned up with PETSc 3 since the matrix
    //        does not have to be set for different linear solvers

    // Initialize PETSc matrix
    MatCreateSeqAIJ(PETSC_COMM_SELF, M, N, PETSC_NULL,
                    reinterpret_cast<int*>(num_nonzeros), A.get());

    // Set some options
    MatSetOption(*A, MAT_KEEP_ZEROED_ROWS, PETSC_TRUE);
    MatZeroEntries(*A);

    // Cleanup
    delete [] num_nonzeros;
  }
  else
  {
    //info("Initializing parallel PETSc matrix (MPIAIJ) of size %d x %d.", M, N);
    //info("Local range is [%d, %d] x [%d, %d].",
    //     row_range.first, row_range.second, col_range.first, col_range.second);

    // Get number of nonzeros for each row from sparsity pattern
    uint* num_nonzeros_diagonal = new uint[m];
    uint* num_nonzeros_off_diagonal = new uint[n];
    assert(num_nonzeros_diagonal);
    assert(num_nonzeros_off_diagonal);
    sparsity_pattern.num_nonzeros_diagonal(num_nonzeros_diagonal);
    sparsity_pattern.num_nonzeros_off_diagonal(num_nonzeros_off_diagonal);

    // Initialize PETSc matrix (MPIAIJ)
    MatCreateMPIAIJ(PETSC_COMM_WORLD,
                    m, n,
                    M, N,
                    PETSC_NULL, reinterpret_cast<int*>(num_nonzeros_diagonal),
                    PETSC_NULL, reinterpret_cast<int*>(num_nonzeros_off_diagonal),
                    A.get());

    // Set some options
    MatSetOption(*A, MAT_KEEP_ZEROED_ROWS, PETSC_TRUE);
    MatZeroEntries(*A);

    // Cleanup
    delete [] num_nonzeros_diagonal;
    delete [] num_nonzeros_off_diagonal;
  }
}
//-----------------------------------------------------------------------------
PETScMatrix* PETScMatrix::copy() const
{
  assert(A);

  // Create copy of PETSc matrix
  boost::shared_ptr<Mat> _Acopy(new Mat, PETScMatrixDeleter());
  MatDuplicate(*A, MAT_COPY_VALUES, _Acopy.get());

  // Create PETScMatrix
  PETScMatrix* Acopy = new PETScMatrix(_Acopy);
  return Acopy;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScMatrix::size(uint dim) const
{
  assert(A);
  int M = 0;
  int N = 0;
  MatGetSize(*A, &M, &N);
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
void PETScMatrix::get(double* block,
                      uint m, const uint* rows,
                      uint n, const uint* cols) const
{
  assert(A);
  MatGetValues(*A,
               static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)),
               static_cast<int>(n), reinterpret_cast<int*>(const_cast<uint*>(cols)),
               block);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const double* block,
                      uint m, const uint* rows,
                      uint n, const uint* cols)
{
  assert(A);
  MatSetValues(*A,
               static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)),
               static_cast<int>(n), reinterpret_cast<int*>(const_cast<uint*>(cols)),
               block, INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScMatrix::add(const double* block,
                      uint m, const uint* rows,
                      uint n, const uint* cols)
{
  assert(A);
  MatSetValues(*A,
               static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)),
               static_cast<int>(n), reinterpret_cast<int*>(const_cast<uint*>(cols)),
               block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScMatrix::axpy(double a, const GenericMatrix& A, bool same_nonzero_pattern)
{
  const PETScMatrix* AA = &A.down_cast<PETScMatrix>();
  assert(this->A);
  assert(AA->mat());
  if (same_nonzero_pattern)
    MatAXPY(*(this->A), a, *AA->mat(), SAME_NONZERO_PATTERN);
  else
    MatAXPY(*(this->A), a, *AA->mat(), DIFFERENT_NONZERO_PATTERN);
}
//-----------------------------------------------------------------------------
void PETScMatrix::getrow(uint row,
                         std::vector<uint>& columns,
                         std::vector<double>& values) const
{
  if (MPI::num_processes() > 1)
    error("PETScMatrix::getrow does not work in parallel.");

  assert(A);

  const int *cols = 0;
  const double *vals = 0;
  int ncols = 0;
  MatGetRow(*A, row, &ncols, &cols, &vals);

  // Assign values to std::vectors
  columns.assign(cols, cols+ncols);
  values.assign(vals, vals+ncols);

  MatRestoreRow(*A, row, &ncols, &cols, &vals);
}
//-----------------------------------------------------------------------------
void PETScMatrix::setrow(uint row,
                         const std::vector<uint>& columns,
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

  // Assign values to arrays
  uint* cols = new uint[n];
  double* vals = new double[n];
  for (uint j = 0; j < n; j++)
  {
    cols[j] = columns[j];
    vals[j] = values[j];
  }

  // Set values
  set(vals, 1, &row, n, cols);

  // Free temporary storage
  delete [] cols;
  delete [] vals;
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero(uint m, const uint* rows)
{
  assert(A);

  IS is = 0;
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), &is);
  PetscScalar null = 0.0;
  MatZeroRowsIS(*A, is, null);
  ISDestroy(is);
}
//-----------------------------------------------------------------------------
void PETScMatrix::ident(uint m, const uint* rows)
{
  assert(A);

  IS is = 0;
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), &is);
  PetscScalar one = 1.0;
  MatZeroRowsIS(*A, is, one);
  ISDestroy(is);
}
//-----------------------------------------------------------------------------
void PETScMatrix::mult(const GenericVector& x, GenericVector& y) const
{
  assert(A);

  const PETScVector& xx = x.down_cast<PETScVector>();
  PETScVector& yy = y.down_cast<PETScVector>();

  if (size(1) != xx.size())
    error("Matrix and vector dimensions don't match for matrix-vector product.");
  yy.resize(size(0));
  MatMult(*A, *xx.vec(), *yy.vec());
}
//-----------------------------------------------------------------------------
void PETScMatrix::transpmult(const GenericVector& x, GenericVector& y) const
{
  assert(A);

  const PETScVector& xx = x.down_cast<PETScVector>();
  PETScVector& yy = y.down_cast<PETScVector>();

  if (size(0) != xx.size())
    error("Matrix and vector dimensions don't match for matrix-vector product.");
  yy.resize(size(1));
  MatMultTranspose(*A, *xx.vec(), *yy.vec());
}
//-----------------------------------------------------------------------------
double PETScMatrix::norm(std::string norm_type) const
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
void PETScMatrix::apply()
{
  assert(A);
  MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero()
{
  assert(A);
  MatZeroEntries(*A);
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator*= (double a)
{
  assert(A);
  MatScale(*A, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator/= (double a)
{
  assert(A);
  MatScale(*A, 1.0 / a);
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
  assert(A.mat());

  // Check for self-assignment
  if (this != &A)
  {
      if (this->A && !this->A.unique())
        error("Cannot assign PETScMatrix with different non-zero pattern because more than one object points to the underlying PETSc object.");
      this->A.reset(new Mat, PETScMatrixDeleter());

      // Duplicate with the same pattern as A.A
      MatDuplicate(*A.mat(), MAT_COPY_VALUES, this->A.get());
    //}
  }
  return *this;
}
//-----------------------------------------------------------------------------
std::string PETScMatrix::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScMatrix not implemented, calling PETSc MatView directly.");

    // FIXME: Maybe this could be an option?
    assert(A);
    if (MPI::num_processes() > 1)
      MatView(*A, PETSC_VIEWER_STDOUT_WORLD);
    else
      MatView(*A, PETSC_VIEWER_STDOUT_SELF);
  }
  else
  {
    s << "<PETScMatrix of size " << size(0) << " x " << size(1) << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& PETScMatrix::factory() const
{
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<Mat> PETScMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------

#endif
