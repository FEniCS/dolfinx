// Copyright (C) 2004-2008 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2007.
// Modified by Andy R. Terrel 2005.
// Modified by Ola Skavhaug 2007.
// Modified by Magnus Vikstrøm 2007-2008.
//
// First added:  2004
// Last changed: 2008-05-15

#ifdef HAS_PETSC

#include <iostream>
#include <sstream>
#include <iomanip>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Array.h>
#include "PETScVector.h"
#include "PETScMatrix.h"
#include "GenericSparsityPattern.h"
#include "SparsityPattern.h"
#include "PETScFactory.h"
#include <dolfin/main/MPI.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const Type type):
    Variable("A", "a sparse matrix"),
    A(0), is_view(false), _type(type)
{
  // Check type
  checkType();
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(Mat A):
    Variable("A", "a sparse matrix"),
    A(A), is_view(true), _type(default_matrix)
{
  // FIXME: get PETSc matrix type and set
  _type = default_matrix;
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(uint M, uint N, Type type):
    Variable("A", "a sparse matrix"),
    A(0), is_view(false), _type(type)
{
  // Check type
  checkType();

  // Create PETSc matrix
  resize(M, N);
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const PETScMatrix& A):
  Variable("A", "PETSc matrix"),
  A(0), is_view(false), _type(A._type)
{
  if (A.mat())
    MatDuplicate(A.mat(), MAT_COPY_VALUES, &(this->A));
}
//-----------------------------------------------------------------------------
PETScMatrix::~PETScMatrix()
{
  if (A && !is_view)
    MatDestroy(A);
}
//-----------------------------------------------------------------------------
void PETScMatrix::resize(uint M, uint N)
{
  // Free previously allocated memory if necessary
  if ( A )
    MatDestroy(A);
  
  // FIXME: maybe 50 should be a parameter?
  // FIXME: it should definitely be a parameter

  // Create a sparse matrix in compressed row format
  if (dolfin::MPI::num_processes() > 1)
  {
    // Create PETSc parallel matrix with a guess for number of diagonal (50 in this case) 
    // and number of off-diagonal non-zeroes (50 in this case).
    // Note that guessing too high leads to excessive memory usage.
    // In order to not waste any memory one would need to specify d_nnz and o_nnz.
    MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, 50, PETSC_NULL, 50, PETSC_NULL, &A);
  }
  else
  {
    // Create PETSc sequential matrix with a guess for number of non-zeroes (50 in thise case)
    MatCreateSeqAIJ(PETSC_COMM_SELF, M, N, 50, PETSC_NULL, &A);

    setType();
    MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
    MatSetFromOptions(A);
  }
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(uint M, uint N, const uint* nz)
{
  // Free previously allocated memory if necessary
  if ( A )
    MatDestroy(A);

  // Create a sparse matrix in compressed row format
  if (dolfin::MPI::num_processes() > 1)
  {
    // Create PETSc parallel matrix with a guess for number of diagonal (50 in this case) 
    // and number of off-diagonal non-zeroes (50 in this case).
    // Note that guessing too high leads to excessive memory usage.
    // In order to not waste any memory one would need to specify d_nnz and o_nnz.
    MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, 50, PETSC_NULL, 50, PETSC_NULL, &A);
    //MatSetFromOptions(A);
    //MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
    //MatZeroEntries(A);
  }
  else
  {
    // Create PETSc sequential matrix with a guess for number of non-zeroes (50 in thise case)
    MatCreate(PETSC_COMM_SELF, &A);
    MatSetSizes(A,  PETSC_DECIDE,  PETSC_DECIDE, M, N);
    setType();
    MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
    MatSetFromOptions(A);
    MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, (int*)nz);
    MatZeroEntries(A);
  }
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(uint M, uint N, const uint* d_nzrow, const uint* o_nzrow)
{
  // Free previously allocated memory if necessary
  if ( A )
    MatDestroy(A);
  // Create PETSc parallel matrix with a guess for number of diagonal (50 in this case) 
  // and number of off-diagonal non-zeroes (50 in this case).
  // Note that guessing too high leads to excessive memory usage.
  // In order to not waste any memory one would need to specify d_nnz and o_nnz.

  MatCreateMPIAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, PETSC_NULL, (int*)d_nzrow, PETSC_NULL, (int*)o_nzrow, &A);
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(const GenericSparsityPattern& sparsity_pattern)
{
  if (dolfin::MPI::num_processes() > 1)
  {
    uint p = dolfin::MPI::process_number();
    const SparsityPattern& spattern = reinterpret_cast<const SparsityPattern&>(sparsity_pattern);
    uint local_size = spattern.numLocalRows(p);
    uint* d_nzrow = new uint[local_size];
    uint* o_nzrow = new uint[local_size];
    spattern.numNonZeroPerRow(p, d_nzrow, o_nzrow);
    init(spattern.size(0), spattern.size(1), d_nzrow, o_nzrow);
    delete [] d_nzrow;
    delete [] o_nzrow;
  }
  else
  {
    const SparsityPattern& spattern = reinterpret_cast<const SparsityPattern&>(sparsity_pattern);
    uint* nzrow = new uint[spattern.size(0)];  
    spattern.numNonZeroPerRow(nzrow);
    init(spattern.size(0), spattern.size(1), nzrow);
    delete [] nzrow;
  }
}
//-----------------------------------------------------------------------------
PETScMatrix* PETScMatrix::copy() const
{
  dolfin_assert(A);

  PETScMatrix* mcopy = new PETScMatrix();
  MatDuplicate(A, MAT_COPY_VALUES, &(mcopy->A));
  return mcopy;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScMatrix::size(uint dim) const
{
  dolfin_assert(A);
  int M = 0;
  int N = 0;
  MatGetSize(A, &M, &N);
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
void PETScMatrix::get(real* block,
                      uint m, const uint* rows,
                      uint n, const uint* cols) const
{
  dolfin_assert(A);
  MatGetValues(A,
               static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)),
               static_cast<int>(n), reinterpret_cast<int*>(const_cast<uint*>(cols)),
               block);
}
//-----------------------------------------------------------------------------
void PETScMatrix::set(const real* block,
                      uint m, const uint* rows,
                      uint n, const uint* cols)
{
  dolfin_assert(A);
  MatSetValues(A,
               static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)),
               static_cast<int>(n), reinterpret_cast<int*>(const_cast<uint*>(cols)),
               block, INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScMatrix::add(const real* block,
                      uint m, const uint* rows,
                      uint n, const uint* cols)
{
  dolfin_assert(A);
  MatSetValues(A,
               static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)),
               static_cast<int>(n), reinterpret_cast<int*>(const_cast<uint*>(cols)),
               block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScMatrix::axpy(real a, const GenericMatrix& A)
{
  const PETScMatrix *AA = &A.down_cast<PETScMatrix>();
  dolfin_assert(this->A);
  dolfin_assert(AA->mat());
  dolfin_assert(sameNonzeroPattern(*AA));
  MatAXPY(this->A,a,AA->mat(),SAME_NONZERO_PATTERN);
}
//-----------------------------------------------------------------------------
void PETScMatrix::getrow(uint row,
                         Array<uint>& columns,
                         Array<real>& values) const
{
  dolfin_assert(A);

  const int *cols = 0;
  const double *vals = 0;
  int ncols = 0;
  MatGetRow(A, row, &ncols, &cols, &vals);
  
  // Assign values to Arrays
  columns.assign(cols, cols+ncols);
  values.assign(vals, vals+ncols);

  MatRestoreRow(A, row, &ncols, &cols, &vals);
}
//-----------------------------------------------------------------------------
void PETScMatrix::setrow(uint row,
                         const Array<uint>& columns,
                         const Array<real>& values)
{
  dolfin_assert(A);

  // Check size of arrays
  if (columns.size() != values.size())
    error("Number of columns and values don't match for setrow() operation.");

  // Handle case n = 0
  const uint n = columns.size();
  if (n == 0)
    return;

  // Assign values to arrays
  uint* cols = new uint[n];
  real* vals = new real[n];
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
  dolfin_assert(A);

  IS is = 0;
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), &is);
  PetscScalar null = 0.0;
  MatZeroRowsIS(A, is, null);
  ISDestroy(is);
}
//-----------------------------------------------------------------------------
void PETScMatrix::ident(uint m, const uint* rows)
{
  dolfin_assert(A);

  IS is = 0;
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), &is);
  PetscScalar one = 1.0;
  MatZeroRowsIS(A, is, one);
  ISDestroy(is);
}
//-----------------------------------------------------------------------------
void PETScMatrix::mult(const GenericVector& x, GenericVector& y, bool transposed) const
{
  dolfin_assert(A);

  const PETScVector& xx = x.down_cast<PETScVector>();  
  PETScVector& yy = y.down_cast<PETScVector>();

  if (transposed)
  { 
    if (size(0) != xx.size()) error("Matrix and vector dimensions don't match for matrix-vector product.");
    yy.resize(size(1));
    MatMultTranspose(A, xx.vec(), yy.vec());
  }
  else {
    if (size(1) != xx.size()) error("Matrix and vector dimensions don't match for matrix-vector product.");
    yy.resize(size(0));
    MatMult(A, xx.vec(), yy.vec());
  }
}
//-----------------------------------------------------------------------------
real PETScMatrix::norm(const NormType type) const
{
  dolfin_assert(A);

  real value = 0.0;
  switch ( type )
  {
    case l1:
      MatNorm(A, NORM_1, &value);
      break;
    case linf:
      MatNorm(A, NORM_INFINITY, &value);
      break;
    case frobenius:
      MatNorm(A, NORM_FROBENIUS, &value);
      break;
    default:
      error("Unknown norm type.");
  }
  return value;
}
//-----------------------------------------------------------------------------
void PETScMatrix::apply()
{
  dolfin_assert(A);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero()
{
  dolfin_assert(A);
  MatZeroEntries(A);
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator*= (real a)
{
  dolfin_assert(A);
  MatScale(A, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScMatrix& PETScMatrix::operator/= (real a)
{
  dolfin_assert(A);
  MatScale(A, 1.0 / a);
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
  dolfin_assert(A.mat());
  // Check for self-assignment
  if (this != &A)
  {
    if (sameNonzeroPattern(A))
      // If the NonzeroPattern is the same we just copy the values
      MatCopy(A.mat(), this->A, SAME_NONZERO_PATTERN);
    
    else
    {
      // Destroy A
      if (this->A && !this->is_view)
	MatDestroy(this->A);
      
      // Creating a new A, with the same pattern as A.A
      MatDuplicate(A.mat(), MAT_COPY_VALUES, &(this->A));
    }
  }
  return *this;
}
//-----------------------------------------------------------------------------
PETScMatrix::Type PETScMatrix::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
void PETScMatrix::disp(uint precision) const
{
  dolfin_assert(A);

  // FIXME: Maybe this could be an option?
  if(MPI::num_processes() > 1)
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  else
    MatView(A, PETSC_VIEWER_STDOUT_SELF);

/*
  const uint M = size(0);
  const uint N = size(1);

  // Sparse output
  for (uint i = 0; i < M; i++)
  {
    std::stringstream line;
    line << std::setiosflags(std::ios::scientific);
    line << std::setprecision(precision);
    
    line << "|";
    
    if ( sparse )
    {
      int ncols = 0;
      const int* cols = 0;
      const double* vals = 0;
      MatGetRow(A, i, &ncols, &cols, &vals);
      for (int pos = 0; pos < ncols; pos++)
      {
	       line << " (" << i << ", " << cols[pos] << ", " << vals[pos] << ")";
      }
      MatRestoreRow(A, i, &ncols, &cols, &vals);
    }
    else
    {
      for (uint j = 0; j < N; j++)
      {
        real value = get(i, j);
        if ( fabs(value) < DOLFIN_EPS )
        value = 0.0;	
        line << " " << value;
      }
    }

    line << "|";
    cout << line.str().c_str() << endl;
  }
*/
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& PETScMatrix::factory() const
{
  return PETScFactory::instance();
}
//-----------------------------------------------------------------------------
Mat PETScMatrix::mat() const
{
  //dolfin_assert(A);
  return A;
}
//-----------------------------------------------------------------------------
void PETScMatrix::setType() 
{
  //dolfin_assert(A);
  MatType mat_type = getPETScType();
  MatSetType(A, mat_type);
}
//-----------------------------------------------------------------------------
void PETScMatrix::checkType()
{
  switch ( _type )
  {
  case default_matrix:
    return;
  case spooles:
    #if !PETSC_HAVE_SPOOLES
      warning("PETSc has not been complied with Spooles. Using default matrix type");
      _type = default_matrix;
    #endif
    return;
  case superlu:
    #if !PETSC_HAVE_SUPERLU
      warning("PETSc has not been complied with Super LU. Using default matrix type");
      _type = default_matrix;
    #endif
    return;
  case umfpack:
    #if !PETSC_HAVE_UMFPACK
      warning("PETSc has not been complied with UMFPACK. Using default matrix type");
      _type = default_matrix;
    #endif
    return;
  default:
    warning("Requested matrix type unknown. Using default.");
    _type = default_matrix;
  }
}
//-----------------------------------------------------------------------------
bool PETScMatrix::sameNonzeroPattern(const PETScMatrix& A) const
{
  // FIXME: This function does not assert for zero pointers. Should it?
  if (this->A==0 or A.mat()==0)
    return false;
  // FIXME: Is this check sufficient? 
  MatInfo this_info, other_info;
  MatGetInfo(this->A,MAT_GLOBAL_SUM,&this_info);
  MatGetInfo(A.mat(),MAT_GLOBAL_SUM,&other_info);
  return this_info.nz_allocated   == other_info.nz_allocated   and \
         this_info.columns_global == other_info.columns_global and \
         this_info.rows_global    == other_info.rows_global;
}
//-----------------------------------------------------------------------------
MatType PETScMatrix::getPETScType() const
{
  switch ( _type )
  {
  case default_matrix:
    if (MPI::num_processes() > 1)
      return MATMPIAIJ;
    else
      return MATSEQAIJ;
  case spooles:
      return MATSEQAIJSPOOLES;
  case superlu:
      return MATSUPERLU;
  case umfpack:
      return MATUMFPACK;
  default:
    return "default";
  }
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const PETScMatrix& A)
{
  stream << "[ PETSc matrix of size " << A.size(0) << " x " << A.size(1) << " ]";
  return stream;
}
//-----------------------------------------------------------------------------

#endif
