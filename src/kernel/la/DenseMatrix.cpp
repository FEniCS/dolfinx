// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/DirectSolver.h>
#include <dolfin/Vector.h>
#include <dolfin/SparseMatrix.h>
#include <dolfin/DenseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix()
{
  values = 0;
  permutation = 0;
  m = 0;
  n = 0;
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(int m, int n)
{
  values = 0;
  this->m = 0;
  this->n = 0;
  permutation = 0;
  
  init(m,n);
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(const DenseMatrix& A)
{
  values = 0;
  permutation = 0;
  this->m = 0;
  this->n = 0;

  init(A.m, A.n);
  
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      values[i][j] = A.values[i][j];
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(const SparseMatrix& A)
{
  values = 0;
  permutation = 0;
  this->m = 0;
  this->n = 0;

  init(A.size(0), A.size(1));
  
  int j;
  real value;
  
  for (int i = 0; i < m; i++)
    for (int pos = 0; pos < A.rowsize(i); pos++){
      value = A(i,j,pos);
      values[i][j] = value;
    }  
}
//-----------------------------------------------------------------------------
DenseMatrix::~DenseMatrix()
{
  clear();
}
//-----------------------------------------------------------------------------
void DenseMatrix::init(int m, int n)
{
  if ( m <= 0 )
    dolfin_error("Number of rows must be positive.");

  if ( n <= 0 )
    dolfin_error("Number of columns must be positive.");

  // Two cases:
  //
  //   1. Already allocated and dimension changes -> reallocate
  //   2. Not allocated -> allocate
  //
  // Otherwise do nothing

  if ( values ) {
    if ( this->m != m || this->n != n ) {
      clear();      
      alloc(m,n);
    }
  }
  else
    alloc(m,n);
}
//-----------------------------------------------------------------------------
void DenseMatrix::clear()
{
  if ( values ) {
    for (int i = 0; i < m; i++)
      delete [] values[i];
    delete [] values;
    values = 0;
  }

  if ( permutation )
    delete [] permutation;
  permutation = 0;
  
  m = 0;
  n = 0;
}
//-----------------------------------------------------------------------------
int DenseMatrix::size(int dim) const
{
  if ( dim == 0 )
    return m;
  else if ( dim == 1 )
    return n;
  
  dolfin_warning1("Illegal matrix dimension: dim = %d.", dim);
  return 0;
}
//-----------------------------------------------------------------------------
int DenseMatrix::size() const
{  
  dolfin_warning("Checking number of non-zero elements in a dense matrix.");
  
  return m*n;
}
//-----------------------------------------------------------------------------
int DenseMatrix::rowsize(int i) const
{
  dolfin_warning("Checking row size for dense matrix.");
  return n;
}
//-----------------------------------------------------------------------------
int DenseMatrix::bytes() const
{
  int bytes = 0;
  
  bytes += sizeof(DenseMatrix);
  bytes += m * sizeof(real *);
  bytes += m * n * sizeof(real);
  bytes += m * sizeof(int);

  return bytes;
}
//-----------------------------------------------------------------------------
real DenseMatrix::operator()(int i, int j) const
{
  return values[i][j];
}
//-----------------------------------------------------------------------------
real* DenseMatrix::operator[](int i)
{
  return values[i];
}
//-----------------------------------------------------------------------------
real DenseMatrix::operator()(int i, int& j, int pos) const
{
  dolfin_warning("Using sparse quick-access operator for dense matrix.");

  j = pos;
  return values[i][j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator=(real a)
{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      values[i][j] = a;
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator=(const DenseMatrix& A)
{
  init(A.m, A.n);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      values[i][j] = A.values[i][j];

  for (int i = 0; i < m; i++)
    permutation[i] = A.permutation[i];
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator=(const SparseMatrix& A)
{
  init(A.m, A.n);
  (*this) = 0.0;

  int j;
  real a;
  for (int i = 0; i < m; i++)
    for (int pos = 0; !A.endrow(i,pos); pos++) {
      real a = A(i,j,pos);
      values[i][j] = a;
    }
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator+=(const DenseMatrix& A)
{
  init(A.m, A.n);

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      values[i][j] += A.values[i][j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator+=(const SparseMatrix& A)
{
  init(A.m, A.n);

  int j;
  real a;
  for (int i = 0; i < m; i++)
    for (int pos = 0; !A.endrow(i,pos); pos++) {
      real a = A(i,j,pos);
      values[i][j] += a;
    }
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator-=(const DenseMatrix& A)
{
  init(A.m, A.n);
  
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      values[i][j] -= A.values[i][j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator-=(const SparseMatrix& A)
{
  init(A.m, A.n);

  int j;
  real a;
  for (int i = 0; i < m; i++)
    for (int pos = 0; !A.endrow(i,pos); pos++) {
      real a = A(i,j,pos);
      values[i][j] -= a;
    }
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator*=(real a)
{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      values[i][j] *= a;
}
//-----------------------------------------------------------------------------
real DenseMatrix::norm() const
{
  real max = 0.0;
  real val;
  
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      if ( (val = fabs(values[i][j])) > max )
	max = val;
  
  return max;
}
//-----------------------------------------------------------------------------
real DenseMatrix::mult(Vector& x, int i) const
{
  if ( n != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  real sum = 0.0;
  for (int j = 0; j < n; j++)
    sum += values[i][j] * x(j);

  return sum;
}
//-----------------------------------------------------------------------------
void DenseMatrix::mult(Vector& x, Vector& Ax) const
{
  if ( n != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  Ax.init(m);
  
  for (int i = 0; i < m; i++)
    Ax(i) = mult(x, i);
}
//-----------------------------------------------------------------------------
void DenseMatrix::resize()
{
  dolfin_warning("Clearing unused elements has no effect for a dense matrix.");
}
//-----------------------------------------------------------------------------
void DenseMatrix::ident(int i)
{
  for (int j = 0; j < n; j++)
    values[i][j] = 0.0;
  values[i][i] = 1.0;
}
//-----------------------------------------------------------------------------
void DenseMatrix::initrow(int i, int rowsize)
{
  dolfin_warning("Specifying number of non-zero elements on a row has no effect for a dense matrix.");
}
//-----------------------------------------------------------------------------
bool DenseMatrix::endrow(int i, int pos) const
{
  dolfin_warning("You probably don't want to use endrow() for a dense matrix.");
  return pos < n;
}
//-----------------------------------------------------------------------------
int DenseMatrix::perm(int i) const
{
  return permutation[i];
}
//-----------------------------------------------------------------------------
void DenseMatrix::show() const 
{
  for (int i = 0; i < m; i++) {
    cout << "| ";
    for (int j = 0; j < n; j++){
      cout << (*this)(i,j) << " ";
    }
    cout << "|" << endl;
  }
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const DenseMatrix& A)
{
  int m = A.size(0);
  int n = A.size(1);
  
  stream << "[ Dense matrix of size " << m << " x " << n << " ]";
}
//-----------------------------------------------------------------------------
void DenseMatrix::alloc(int m, int n)
{
  // Use with caution. Only for internal use.

  values = new (real *)[m];
  permutation = new int[m];
  
  for (int i = 0; i < m; i++)
    values[i] = new real[n];
  
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      values[i][j] = 0.0;
  
  for (int i = 0; i < m; i++)
    permutation[i] = i;  

  this->m = m;
  this->n = n;
}
//-----------------------------------------------------------------------------
real DenseMatrix::read(int i, int j) const
{
  return values[i][j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::write(int i, int j, real value)
{
  values[i][j] = value;
}
//-----------------------------------------------------------------------------
void DenseMatrix::add(int i, int j, real value)
{
  values[i][j] += value;
}
//-----------------------------------------------------------------------------
void DenseMatrix::sub(int i, int j, real value)
{
  values[i][j] -= value;
}
//-----------------------------------------------------------------------------
void DenseMatrix::mult(int i, int j, real value)
{
  values[i][j] *= value;
}
//-----------------------------------------------------------------------------
void DenseMatrix::div(int i, int j, real value)
{
  values[i][j] /= value;
}
//-----------------------------------------------------------------------------
real** DenseMatrix::getvalues()
{
  return values;
}
//-----------------------------------------------------------------------------
int* DenseMatrix::getperm()
{
  return permutation;
}
//-----------------------------------------------------------------------------
