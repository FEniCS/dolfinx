// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <sstream>
#include <iomanip>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Matrix::Matrix() : A(0)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
Matrix::Matrix(uint M, uint N) : A(0)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  init(M, N);
}
//-----------------------------------------------------------------------------
Matrix::Matrix(const Matrix& B) : A(0)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  init(B.size(0), B.size(1));
  
  uint M = B.size(0);
  uint N = B.size(1);

  // FIXME: Use PETSc function to copy
  for(uint i = 0; i < M; i++)
  {
    for(uint j = 0; j < N; j++)
    {
      setval(i, j, B(i, j));
    }
  }
}
//-----------------------------------------------------------------------------
Matrix::~Matrix()
{
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void Matrix::init(uint M, uint N)
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
  MatCreateSeqAIJ(PETSC_COMM_SELF, M, N, 50, PETSC_NULL, &A);
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
  MatSetFromOptions(A);
}
//-----------------------------------------------------------------------------
void Matrix::init(uint M, uint N, uint bs)
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
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
  MatSetFromOptions(A);
}
//-----------------------------------------------------------------------------
void Matrix::init(uint M, uint N, uint bs, uint mnc)
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
  MatSetOption(A, MAT_KEEP_ZEROED_ROWS);
  MatSetFromOptions(A);
}
//-----------------------------------------------------------------------------
dolfin::uint Matrix::size(uint dim) const
{
  int M = 0;
  int N = 0;
  MatGetSize(A, &M, &N);
  return (dim == 0 ? M : N);
}
//-----------------------------------------------------------------------------
Matrix& Matrix::operator= (real zero)
{
  if ( zero != 0.0 )
    dolfin_error("Argument must be zero.");
  MatZeroEntries(A);
  return *this;
}
//-----------------------------------------------------------------------------
void Matrix::add(const real block[],
		    const int rows[], int m,
		    const int cols[], int n)
{
  MatSetValues(A, m, rows, n, cols, block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void Matrix::ident(const int rows[], int m)
{
  IS is = 0;
  ISCreateGeneral(PETSC_COMM_SELF, m, rows, &is);
  real one = 1.0;
  MatZeroRows(A, is, &one);
  ISDestroy(is);
}
//-----------------------------------------------------------------------------
void Matrix::mult(const Vector& x, Vector& Ax) const
{
  MatMult(A, x.vec(), Ax.vec());
}
//-----------------------------------------------------------------------------
void Matrix::apply()
{
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
Mat Matrix::mat()
{
  return A;
}
//-----------------------------------------------------------------------------
const Mat Matrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
void Matrix::disp(bool sparse, int precision) const
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
    line << "| ";

    for (uint j = 0; j < N; j++)
    {
      real value = getval(i, j);
      if ( fabs(value) < DOLFIN_EPS )
	value = 0.0;

      line << std::setw(precision + 3) << value << " ";
    }
    line << "|";

    cout << line.str().c_str() << endl;
  }
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const Matrix& A)
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
Matrix::Element Matrix::operator()(uint i, uint j)
{
  Element element(i, j, *this);

  return element;
}
//-----------------------------------------------------------------------------
real Matrix::operator() (uint i, uint j) const
{
  return getval(i, j);
}
//-----------------------------------------------------------------------------
real Matrix::getval(uint i, uint j) const
{
  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);

  dolfin_assert(A);
  PetscScalar val;
  MatGetValues(A, 1, &ii, 1, &jj, &val);

  return val;
}
//-----------------------------------------------------------------------------
void Matrix::setval(uint i, uint j, const real a)
{
  MatSetValue(A, i, j, a, INSERT_VALUES);

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
void Matrix::addval(uint i, uint j, const real a)
{
  MatSetValue(A, i, j, a, ADD_VALUES);

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
// Matrix::Element
//-----------------------------------------------------------------------------
Matrix::Element::Element(uint i, uint j, Matrix& A) : i(i), j(j), A(A)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Matrix::Element::operator real() const
{
  return A.getval(i, j);
}
//-----------------------------------------------------------------------------
const Matrix::Element& Matrix::Element::operator=(const real a)
{
  A.setval(i, j, a);

  return *this;
}
//-----------------------------------------------------------------------------
const Matrix::Element& Matrix::Element::operator=(const Element& e)
{
  A.setval(i, j, e.A.getval(e.i, e.j));

  return *this;
}
//-----------------------------------------------------------------------------
const Matrix::Element& Matrix::Element::operator+=(const real a)
{
  A.addval(i, j, a);

  return *this;
}
//-----------------------------------------------------------------------------
const Matrix::Element& Matrix::Element::operator-=(const real a)
{
  A.addval(i, j, -a);

  return *this;
}
//-----------------------------------------------------------------------------
const Matrix::Element& Matrix::Element::operator*=(const real a)
{
  const real val = A.getval(i, j) * a;
  A.setval(i, j, val);
  
  return *this;
}
//-----------------------------------------------------------------------------
