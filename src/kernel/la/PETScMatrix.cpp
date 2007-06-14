// Copyright (C) 2004-2007 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2007.
// Modified by Andy R. Terrel 2005.
//
// First added:  2004
// Last changed: 2007-04-16

#ifdef HAVE_PETSC_H

#include <iostream>
#include <sstream>
#include <iomanip>
#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/PETScManager.h>
#include <dolfin/PETScVector.h>
#include <dolfin/PETScMatrix.h>
#include <dolfin/SparsityPattern.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(const Type type)
  : GenericMatrix(), 
    Variable("A", "a sparse matrix"),
    A(0), _type(type)
{
  // Initialize PETSc
  PETScManager::init();

  // Check type
  checkType();
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(Mat A)
  : GenericMatrix(),
    Variable("A", "a sparse matrix"),
    A(A), _type(default_matrix)
{
  // Initialize PETSc
  PETScManager::init();

  // FIXME: get PETSc matrix type and set
  _type = default_matrix;
}
//-----------------------------------------------------------------------------
PETScMatrix::PETScMatrix(uint M, uint N, Type type)
  : GenericMatrix(), 
    Variable("A", "a sparse matrix"),
    A(0),  _type(type)
{
  // Initialize PETSc
  PETScManager::init();

  // Check type
  checkType();

  // Create PETSc matrix
  init(M, N);
}
//-----------------------------------------------------------------------------
PETScMatrix::~PETScMatrix()
{
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(uint M, uint N)
{
  // Free previously allocated memory if necessary
  if ( A )
    MatDestroy(A);
  
  // FIXME: maybe 50 should be a parameter?
  // FIXME: it should definitely be a parameter

  // Create a sparse matrix in compressed row format
  MatCreateSeqAIJ(PETSC_COMM_SELF, M, N, 50, PETSC_NULL, &A);
  setType();
  MatSetFromOptions(A);
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(uint M, uint N, const uint nz[])
{
  // Free previously allocated memory if necessary
  if ( A )
    MatDestroy(A);

  // Create a sparse matrix in compressed row format
  MatCreate(PETSC_COMM_SELF, &A);
  MatSetSizes(A,  PETSC_DECIDE,  PETSC_DECIDE, M, N);
  setType();
  MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, (int*)nz);
  MatSetFromOptions(A);
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
  MatZeroEntries(A);
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(const uint M, const uint N, const uint bs, const uint nz)
{
  // Free previously allocated memory if necessary
  if ( A )
    MatDestroy(A);
  
  // Creates a sparse matrix in block AIJ (block compressed row) format.
  // Given blocksize bs, and max no connectivity mnc.  
  MatCreateSeqBAIJ(PETSC_COMM_SELF, bs, bs*M, bs*N, nz, PETSC_NULL, &A);
  MatSetFromOptions(A);
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
  MatZeroEntries(A);
}
//-----------------------------------------------------------------------------
void PETScMatrix::init(const SparsityPattern& sparsity_pattern)
{
  uint* nzrow = new uint[sparsity_pattern.size(0)];  
  sparsity_pattern.numNonZeroPerRow(nzrow);
  init(sparsity_pattern.size(0), sparsity_pattern.size(1), nzrow);
  delete [] nzrow;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScMatrix::size(const uint dim) const
{
  int M = 0;
  int N = 0;
  MatGetSize(A, &M, &N);
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScMatrix::nz(const uint row) const
{
  // FIXME: this can probably be done better
  int ncols = 0;
  const int* cols = 0;
  const double* vals = 0;
  MatGetRow(A,     static_cast<int>(row), &ncols, &cols, &vals);
  MatRestoreRow(A, static_cast<int>(row), &ncols, &cols, &vals);

  return ncols;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScMatrix::nzsum() const
{
  uint M = size(0);
  uint sum = 0;
  for (uint i = 0; i < M; i++)
    sum += nz(i);

  return sum;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScMatrix::nzmax() const
{
  uint M = size(0);
  uint max = 0;
  for (uint i = 0; i < M; i++)
    max = std::max(max, nz(i));

  return max;
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
void PETScMatrix::getRow(const uint i, int& ncols, Array<int>& columns, 
                         Array<real>& values) const
{
  const int *cols = 0;
  const double *vals = 0;
  MatGetRow(A, i, &ncols, &cols, &vals);
  
  // Assign values to Arrays
  columns.assign(cols, cols+ncols);
  values.assign(vals, vals+ncols);

  MatRestoreRow(A, i, &ncols, &cols, &vals);
}
//-----------------------------------------------------------------------------
void PETScMatrix::ident(const uint rows[], uint m)
{
  IS is = 0;
  ISCreateGeneral(PETSC_COMM_SELF, static_cast<int>(m), reinterpret_cast<int*>(const_cast<uint*>(rows)), &is);
  PetscScalar one = 1.0;
  MatZeroRowsIS(A, is, one);
  ISDestroy(is);
}
//-----------------------------------------------------------------------------
void PETScMatrix::mult(const PETScVector& x, PETScVector& Ax) const
{
  MatMult(A, x.vec(), Ax.vec());
}
//-----------------------------------------------------------------------------
real PETScMatrix::mult(const PETScVector& x, const uint row) const
{
  // FIXME: Temporary fix (assumes uniprocessor case)

  int ncols = 0;
  const int* cols = 0;
  const double* Avals = 0;
  double* xvals = 0;
  MatGetRow(A, static_cast<int>(row), &ncols, &cols, &Avals);
  VecGetArray(x.x, &xvals);

  real sum = 0.0;
  for (int i = 0; i < ncols; i++)
    sum += Avals[i] * xvals[cols[i]];

  MatRestoreRow(A, static_cast<int>(row), &ncols, &cols, &Avals);
  VecRestoreArray(x.x, &xvals);

  return sum;
}
//-----------------------------------------------------------------------------
real PETScMatrix::mult(const real x[], const uint row) const
{
  // FIXME: Temporary fix (assumes uniprocessor case)

  int ncols = 0;
  const int* cols = 0;
  const double* Avals = 0;
  MatGetRow(A, static_cast<int>(row), &ncols, &cols, &Avals);

  real sum = 0.0;
  for (int i = 0; i < ncols; i++)
    sum += Avals[i] * x[cols[i]];

  MatRestoreRow(A, static_cast<int>(row), &ncols, &cols, &Avals);

  return sum;
}
//-----------------------------------------------------------------------------
void PETScMatrix::lump(PETScVector& m) const
{
  m.init(size(0));
  PETScVector one(m);
  one = 1.0;
  mult(one, m);   
}
//-----------------------------------------------------------------------------
real PETScMatrix::norm(const Norm type) const
{
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
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void PETScMatrix::zero()
{
  MatZeroEntries(A);
}
//-----------------------------------------------------------------------------
PETScMatrix::Type PETScMatrix::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
Mat PETScMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
void PETScMatrix::disp(uint precision) const
{
  // FIXME: Maybe this could be an option?
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
LogStream& dolfin::operator<< (LogStream& stream, const PETScMatrix& A)
{
  // Check if matrix has been defined
  if ( !A.A )
  {
    stream << "[ PETSc matrix (empty) ]";
    return stream;
  }

  MatType type = 0;
  MatGetType(A.mat(), &type);
  int m = A.size(0);
  int n = A.size(1);
  stream << "[ PETSc matrix (type " << type << ") of size "
	  << m << " x " << n << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
void PETScMatrix::setType() 
{
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
MatType PETScMatrix::getPETScType() const
{
  switch ( _type )
  {
  case default_matrix:
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

#endif
