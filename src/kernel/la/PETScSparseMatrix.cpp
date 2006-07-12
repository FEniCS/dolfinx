// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005,2006.
// Modified by Andy R. Terrel 2005.
//
// First added:  2004
// Last changed: 2006-05-29

#ifdef HAVE_PETSC_H

#include <iostream>
#include <sstream>
#include <iomanip>
#include <dolfin/dolfin_log.h>
#include <dolfin/Array.h>
#include <dolfin/PETScManager.h>
#include <dolfin/PETScVector.h>
#include <dolfin/PETScSparseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScSparseMatrix::PETScSparseMatrix()
  : GenericMatrix(),
    Variable("A", "a sparse matrix"),
    A(0), _type(default_matrix)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
PETScSparseMatrix::PETScSparseMatrix(Type type)
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
PETScSparseMatrix::PETScSparseMatrix(Mat A)
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
PETScSparseMatrix::PETScSparseMatrix(uint M, uint N)
  : GenericMatrix(), 
    Variable("A", "a sparse matrix"),
    A(0), _type(default_matrix)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  init(M, N);
}
//-----------------------------------------------------------------------------
PETScSparseMatrix::PETScSparseMatrix(uint M, uint N, Type type)
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
PETScSparseMatrix::PETScSparseMatrix(const PETScSparseMatrix& B)
  : GenericMatrix(), 
    Variable("A", "a sparse matrix"), 
    A(0), _type(B._type)
{
  // Initialize PETSc
  PETScManager::init();

  // Create new PETSc matrix which is a copy of B
  MatDuplicate(B.A, MAT_COPY_VALUES, &A);  
}
//-----------------------------------------------------------------------------
PETScSparseMatrix::~PETScSparseMatrix()
{
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::init(uint M, uint N)
{
  // Free previously allocated memory if necessary
  if ( A )
  {
    if ( M == size(0) && N == size(1) )
      return;
    else
      MatDestroy(A);
  }
  
  // FIXME: maybe 50 should be a parameter?
  // FIXME: it should definitely be a parameter

  // Create a sparse matrix in compressed row format
  MatCreateSeqAIJ(PETSC_COMM_SELF, M, N, 50, PETSC_NULL, &A);
  MatSetFromOptions(A);
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::init(uint M, uint N, uint nz)
{
  // Free previously allocated memory if necessary
  if ( A )
  {
    if ( M == size(0) && N == size(1) )
      return;
    else
      MatDestroy(A);
  }

  // Create a sparse matrix in compressed row format
  MatCreate(PETSC_COMM_SELF, &A);
  MatSetSizes(A,  PETSC_DECIDE,  PETSC_DECIDE, M, N);
  setType();
  MatSeqAIJSetPreallocation(A, nz, PETSC_NULL);
  MatSetFromOptions(A);
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::init(uint M, uint N, uint bs, uint nz)
{
  // Free previously allocated memory if necessary
  if ( A )
  {
    if ( M == size(0) && N == size(1) )
      return;
    else
      MatDestroy(A);
  }
  
  // Creates a sparse matrix in block AIJ (block compressed row) format.
  // Given blocksize bs, and max no connectivity mnc.  
  MatCreateSeqBAIJ(PETSC_COMM_SELF, bs, bs*M, bs*N, nz, PETSC_NULL, &A);
  MatSetFromOptions(A);
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScSparseMatrix::size(uint dim) const
{
  int M = 0;
  int N = 0;
  MatGetSize(A, &M, &N);
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
dolfin::uint PETScSparseMatrix::nz(uint row) const
{
  // FIXME: this can probably be done better
  int ncols = 0;
  const int* cols = 0;
  const double* vals = 0;
  MatGetRow(A, static_cast<int>(row), &ncols, &cols, &vals);
  MatRestoreRow(A, static_cast<int>(row), &ncols, &cols, &vals);

  return ncols;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScSparseMatrix::nzsum() const
{
  uint M = size(0);
  uint sum = 0;
  for (uint i = 0; i < M; i++)
    sum += nz(i);

  return sum;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScSparseMatrix::nzmax() const
{
  uint M = size(0);
  uint max = 0;
  for (uint i = 0; i < M; i++)
    max = std::max(max, nz(i));

  return max;
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::set(const real block[],
		       const int rows[], int m,
		       const int cols[], int n)
{
  MatSetValues(A, m, rows, n, cols, block, INSERT_VALUES);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::add(const real block[],
		       const int rows[], int m,
		       const int cols[], int n)
{
  MatSetValues(A, m, rows, n, cols, block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::getRow(const uint i, int& ncols, Array<int>& columns, 
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
void PETScSparseMatrix::ident(const int rows[], int m)
{
  IS is = 0;
  ISCreateGeneral(PETSC_COMM_SELF, m, rows, &is);
  PetscScalar one = 1.0;
  MatZeroRowsIS(A, is, one);
  ISDestroy(is);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::mult(const PETScVector& x, PETScVector& Ax) const
{
  MatMult(A, x.vec(), Ax.vec());
}
//-----------------------------------------------------------------------------
real PETScSparseMatrix::mult(const PETScVector& x, uint row) const
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
real PETScSparseMatrix::mult(const real x[], uint row) const
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
void PETScSparseMatrix::lump(PETScVector& m) const
{
  m.init(size(0));
  PETScVector one(m);
  one = 1.0;
  mult(one, m);   
}
//-----------------------------------------------------------------------------
real PETScSparseMatrix::norm(Norm type) const
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
    dolfin_error("Unknown norm type.");
  }
  
  return value;
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::apply()
{
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::zero()
{
  MatZeroEntries(A);
}
//-----------------------------------------------------------------------------
PETScSparseMatrix::Type PETScSparseMatrix::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
Mat PETScSparseMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::disp(bool sparse, int precision) const
{
  // FIXME: Maybe this could be an option?
  //MatView(A, PETSC_VIEWER_STDOUT_SELF);

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
        real value = getval(i, j);
        if ( fabs(value) < DOLFIN_EPS )
        value = 0.0;	
        line << " " << value;
      }
    }

    line << "|";
    cout << line.str().c_str() << endl;
  }
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const PETScSparseMatrix& A)
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
PETScSparseMatrixElement PETScSparseMatrix::operator()(uint i, uint j)
{
  PETScSparseMatrixElement element(i, j, *this);

  return element;
}
//-----------------------------------------------------------------------------
real PETScSparseMatrix::operator() (uint i, uint j) const
{
  return getval(i, j);
}
//-----------------------------------------------------------------------------
real PETScSparseMatrix::getval(uint i, uint j) const
{
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);

  dolfin_assert(A);
  PetscScalar val;
  MatGetValues(A, 1, &ii, 1, &jj, &val);

  return val;
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::setval(uint i, uint j, const real a)
{
  MatSetValue(A, i, j, a, INSERT_VALUES);

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::addval(uint i, uint j, const real a)
{
  MatSetValue(A, i, j, a, ADD_VALUES);

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::setType() 
{
  MatType mat_type = getPETScType();
  MatSetType(A, mat_type);
}
//-----------------------------------------------------------------------------
void PETScSparseMatrix::checkType()
{
  switch ( _type )
  {
  case spooles:
    #if !PETSC_HAVE_SPOOLES
      dolfin_warning("PETSc has not been complied with Spooles. Using default matrix type");
      _type = default_matrix;
    #endif
    return;
  case superlu:
    #if !PETSC_HAVE_SUPERLU
      dolfin_warning("PETSc has not been complied with Super LU. Using default matrix type");
      _type = default_matrix;
    #endif
    return;
  case umfpack:
    #if !PETSC_HAVE_UMFPACK
      dolfin_warning("PETSc has not been complied with UMFPACK. Using default matrix type");
      _type = default_matrix;
    #endif
    return;
  default:
    dolfin_warning("Requested matrix type unknown. Using default.");
    _type = default_matrix;
  }
}
//-----------------------------------------------------------------------------
MatType PETScSparseMatrix::getPETScType() const
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
// PETScSparseMatrixElement
//-----------------------------------------------------------------------------
PETScSparseMatrixElement::PETScSparseMatrixElement(uint i, uint j, 
      PETScSparseMatrix& A) : i(i), j(j), A(A)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScSparseMatrixElement::PETScSparseMatrixElement(const PETScSparseMatrixElement& e) 
      : i(i), j(j), A(A)
{
}
//-----------------------------------------------------------------------------
PETScSparseMatrixElement::operator real() const
{
  return A.getval(i, j);
}
//-----------------------------------------------------------------------------
const PETScSparseMatrixElement& PETScSparseMatrixElement::operator=(const real a)
{
  A.setval(i, j, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScSparseMatrixElement& PETScSparseMatrixElement::operator=(const PETScSparseMatrixElement& e)
{
  A.setval(i, j, e.A.getval(e.i, e.j));
  return *this;
}
//-----------------------------------------------------------------------------
const PETScSparseMatrixElement& PETScSparseMatrixElement::operator+=(const real a)
{
  A.addval(i, j, a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScSparseMatrixElement& PETScSparseMatrixElement::operator-=(const real a)
{
  A.addval(i, j, -a);
  return *this;
}
//-----------------------------------------------------------------------------
const PETScSparseMatrixElement& PETScSparseMatrixElement::operator*=(const real a)
{
  const real val = A.getval(i, j) * a;
  A.setval(i, j, val);
  return *this;
}
//-----------------------------------------------------------------------------

#endif
