// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <sstream>
#include <iomanip>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/NewVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewMatrix::NewMatrix()
{
  // Initialize PETSc
  PETScManager::init();

  // Don't initialize the matrix
  A = 0;
}
//-----------------------------------------------------------------------------
NewMatrix::NewMatrix(uint M, uint N)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  A = 0;
  init(M, N);
}
//-----------------------------------------------------------------------------
NewMatrix::NewMatrix(const Matrix& B)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  A = 0;
  init(B.size(0), B.size(1));
  
  uint M = B.size(0);
  uint N = B.size(1);

  for(uint i = 0; i < M; i++)
  {
    for(uint j = 0; j < N; j++)
    {
      setval(i, j, B(i, j));
    }
  }
}
//-----------------------------------------------------------------------------
NewMatrix::~NewMatrix()
{
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void NewMatrix::init(uint M, uint N)
{
  // Free previously allocated memory if necessary
  if ( A )
  {
    if ( M == size(0) && N == size(1) )
      return;
    else
      MatDestroy(A);
  }
  
  //  MatCreate(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, &A);
  //  MatSetFromOptions(A);

  // Creates a sparse matrix in block AIJ (block compressed row) format.
  // Assuming blocksize bs=1, and max no connectivity = 50 
  MatCreateSeqBAIJ(PETSC_COMM_SELF, 1, M, N, 50, PETSC_NULL, &A);
}
//-----------------------------------------------------------------------------
void NewMatrix::init(uint M, uint N, uint bs)
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
  // Given blocksize bs, and assuming max no connectivity = 50. 
  MatCreateSeqBAIJ(PETSC_COMM_SELF, bs, bs*M, bs*N, 50, PETSC_NULL, &A);
}
//-----------------------------------------------------------------------------
void NewMatrix::init(uint M, uint N, uint bs, uint mnc)
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
  MatCreateSeqBAIJ(PETSC_COMM_SELF, bs, bs*M, bs*N, mnc, PETSC_NULL, &A);
}
//-----------------------------------------------------------------------------
dolfin::uint NewMatrix::size(uint dim) const
{
  int M = 0;
  int N = 0;
  MatGetSize(A, &M, &N);
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
NewMatrix& NewMatrix::operator= (real zero)
{
  if ( zero != 0.0 )
    dolfin_error("Argument must be zero.");
  MatZeroEntries(A);
  return *this;
}
//-----------------------------------------------------------------------------
void NewMatrix::add(const real block[],
		    const int rows[], int m,
		    const int cols[], int n)
{
  MatSetValues(A, m, rows, n, cols, block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void NewMatrix::ident(const int rows[], int m)
{
  IS is = 0;
  ISCreateGeneral(PETSC_COMM_SELF, m, rows, &is);
  real one = 1.0;
  MatZeroRows(A, is, &one);
  ISDestroy(is);
}
//-----------------------------------------------------------------------------
void NewMatrix::mult(const NewVector& x, NewVector& Ax) const
{
  MatMult(A, x.vec(), Ax.vec());
}
//-----------------------------------------------------------------------------
void NewMatrix::apply()
{
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
Mat NewMatrix::mat()
{
  return A;
}
//-----------------------------------------------------------------------------
const Mat NewMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
void NewMatrix::disp(bool sparse, int precision) const
{
  // Use PETSc sparse output as default
  if ( sparse )
  {
    MatView(A, PETSC_VIEWER_STDOUT_SELF);
    return;
  }

  // Dense output
  const uint M = size(0);
  const uint N = size(1);
  for (uint i = 0; i < M; i++)
  {
    std::stringstream line;

    line << std::setprecision(precision);

    for (uint j = 0; j < N; j++)
    {
      real value = getval(i, j);
      if ( fabs(value) < DOLFIN_EPS )
	value = 0.0;

      line << std::setw(precision + 1) << value << " ";
    }
    cout << line.str().c_str() << endl;
  }  
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const NewMatrix& A)
{
  MatType type = 0;
  MatGetType(A.mat(), &type);
  int m = A.size(0);
  int n = A.size(1);
  stream << "[ PETSc matrix (type " << type << ") of size "
	 << m << " x " << n << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
NewMatrix::Element NewMatrix::operator()(uint i, uint j)
{
  Element element(i, j, *this);

  return element;
}
//-----------------------------------------------------------------------------
real NewMatrix::getval(uint i, uint j) const
{
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);

  PetscScalar val;
  MatGetValues(A, 1, &ii, 1, &jj, &val);

  return val;
}
//-----------------------------------------------------------------------------
void NewMatrix::setval(uint i, uint j, const real a)
{
  MatSetValue(A, i, j, a, INSERT_VALUES);

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void NewMatrix::addval(uint i, uint j, const real a)
{
  MatSetValue(A, i, j, a, ADD_VALUES);

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
// NewMatrix::Element
//-----------------------------------------------------------------------------
NewMatrix::Element::Element(uint i, uint j, NewMatrix& A) : i(i), j(j), A(A)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewMatrix::Element::operator real() const
{
  return A.getval(i, j);
}
//-----------------------------------------------------------------------------
const NewMatrix::Element& NewMatrix::Element::operator=(const real a)
{
  A.setval(i, j, a);

  return *this;
}
//-----------------------------------------------------------------------------
const NewMatrix::Element& NewMatrix::Element::operator+=(const real a)
{
  A.addval(i, j, a);

  return *this;
}
//-----------------------------------------------------------------------------
const NewMatrix::Element& NewMatrix::Element::operator-=(const real a)
{
  A.addval(i, j, -a);

  return *this;
}
//-----------------------------------------------------------------------------
const NewMatrix::Element& NewMatrix::Element::operator*=(const real a)
{
  const real val = A.getval(i, j) * a;
  A.setval(i, j, val);
  
  return *this;
}
//-----------------------------------------------------------------------------
