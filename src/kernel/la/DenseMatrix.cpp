// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson, 2003.
// Modified by Karin Kraft, 2004.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/DirectSolver.h>
#include <dolfin/Vector.h>
#include <dolfin/SparseMatrix.h>
#include <dolfin/DenseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix():
  values(0),
  permutation(0)
{
  m = 0;
  n = 0;
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(unsigned int m, unsigned int n):
  values(0),
  permutation(0)
{
  this->m = 0;
  this->n = 0;
    
  init(m,n);
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(const DenseMatrix& A):
  values(0),
  permutation(0)
{
  this->m = 0;
  this->n = 0;

  init(A.m, A.n);
  
  for (unsigned int i = 0; i < n; i++)
    for (unsigned int j = 0; j < n; j++)
      values[i][j] = A.values[i][j];
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(const SparseMatrix& A) :
  values(0),
  permutation(0)
{

  this->m = 0;
  this->n = 0;

  init(A.size(0), A.size(1));
  
  unsigned int j;
  real value;
  
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; pos < A.rowsize(i); pos++) {
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
void DenseMatrix::init(unsigned int m, unsigned int n)
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

  clearperm();
}
//-----------------------------------------------------------------------------
void DenseMatrix::clear()
{
  if ( values ) {
    for (unsigned int i = 0; i < m; i++)
      delete [] values[i];
    delete [] values;
    values = 0;
  }

  clearperm();
  
  m = 0;
  n = 0;
}
//-----------------------------------------------------------------------------
unsigned int DenseMatrix::size(unsigned int dim) const
{
  if ( dim == 0 )
    return m;
  else if ( dim == 1 )
    return n;
  
  dolfin_warning1("Illegal matrix dimension: dim = %d.", dim);
  return 0;
}
//-----------------------------------------------------------------------------
unsigned int DenseMatrix::size() const
{  
  dolfin_warning("Checking number of non-zero elements in a dense matrix.");
  
  return m*n;
}
//-----------------------------------------------------------------------------
unsigned int DenseMatrix::rowsize(unsigned int i) const
{
  dolfin_warning("Checking row size for dense matrix.");
  return n;
}
//-----------------------------------------------------------------------------
unsigned int DenseMatrix::bytes() const
{
  unsigned int bytes = 0;
  
  bytes += sizeof(DenseMatrix);
  bytes += m * sizeof(real *);
  bytes += m * n * sizeof(real);
  bytes += m * sizeof(unsigned int);

  return bytes;
}
//-----------------------------------------------------------------------------
real DenseMatrix::operator()(unsigned int i, unsigned int j) const
{
  return values[i][j];
}
//-----------------------------------------------------------------------------
real* DenseMatrix::operator[](unsigned int i)
{
  return values[i];
}
//-----------------------------------------------------------------------------
real DenseMatrix::operator()(unsigned int i, unsigned int& j, unsigned int pos) const
{
  dolfin_warning("Using sparse quick-access operator for dense matrix.");

  j = pos;
  return values[i][j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator=(real a)
{
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      values[i][j] = a;
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator=(const DenseMatrix& A)
{
  init(A.m, A.n);

  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      values[i][j] = A.values[i][j];

  if ( A.permutation ) {
    initperm();
    for (unsigned int i = 0; i < m; i++)
      permutation[i] = A.permutation[i];
  }
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator=(const SparseMatrix& A)
{
  init(A.m, A.n);
  (*this) = 0.0;

  unsigned int j;
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; !A.endrow(i,pos); pos++) {
      real a = A(i,j,pos);
      values[i][j] = a;
    }

  clearperm();
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator+=(const DenseMatrix& A)
{
  init(A.m, A.n);

  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      values[i][j] += A.values[i][j];
  
  clearperm();
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator+=(const SparseMatrix& A)
{
  init(A.m, A.n);

  unsigned int j;
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; !A.endrow(i,pos); pos++) {
      real a = A(i,j,pos);
      values[i][j] += a;
    }

  clearperm();
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator-=(const DenseMatrix& A)
{
  init(A.m, A.n);
  
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      values[i][j] -= A.values[i][j];

  clearperm();
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator-=(const SparseMatrix& A)
{
  init(A.m, A.n);

  unsigned int j;
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; !A.endrow(i,pos); pos++) {
      real a = A(i,j,pos);
      values[i][j] -= a;
    }

  clearperm();
}
//-----------------------------------------------------------------------------
void DenseMatrix::operator*=(real a)
{
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      values[i][j] *= a;

  clearperm();
}
//-----------------------------------------------------------------------------
real DenseMatrix::norm() const
{
  real max = 0.0;
  real val;
  
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      if ( (val = fabs(values[i][j])) > max )
	max = val;
  
  return max;
}
//-----------------------------------------------------------------------------
real DenseMatrix::mult(const Vector& x, unsigned int i) const
{
  if ( n != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  real sum = 0.0;
  for (unsigned int j = 0; j < n; j++)
    sum += values[i][j] * x(j);

  return sum;
}
//-----------------------------------------------------------------------------
void DenseMatrix::mult(const Vector& x, Vector& Ax) const
{
  if ( n != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  Ax.init(m);
  
  for (unsigned int i = 0; i < m; i++)
    Ax(i) = mult(x, i);
}
//-----------------------------------------------------------------------------
void DenseMatrix::multt(const Vector& x, Vector& Ax) const
{
  if ( m != x.size() )
    dolfin_error("Matrix dimensions don't match.");
  
  Ax.init(n);
  Ax = 0.0;
                                                                                                                                                            
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      Ax(j) += values[i][j] * x(i);
}
//-----------------------------------------------------------------------------
real DenseMatrix::multrow(const Vector& x, unsigned int i) const
{
  return mult(x,i);
}
//-----------------------------------------------------------------------------
real DenseMatrix::multcol(const Vector& x, unsigned int j) const
{
  if ( m != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  real sum = 0.0;
  for (unsigned int i = 0; i < m; i++)
    sum += values[i][j] * x(j);

  return sum;
}
//-----------------------------------------------------------------------------
void DenseMatrix::resize()
{
  dolfin_warning("Clearing unused elements has no effect for a dense matrix.");
}
//-----------------------------------------------------------------------------
void DenseMatrix::ident(unsigned int i)
{
  for (unsigned int j = 0; j < n; j++)
    values[i][j] = 0.0;
  values[i][i] = 1.0;
}
//-----------------------------------------------------------------------------
void DenseMatrix::lump(Vector& a) const
{
  a.init(m);

  for (unsigned int i = 0; i < m; i++)
  {
    a(i) = 0.0;
    for (unsigned int j = 0; j < n; j++)
      a(i) += values[i][j];
  }
}
//-----------------------------------------------------------------------------
void DenseMatrix::addrow()
{
  real** new_values = new real * [m+1];

  for (unsigned int i = 0; i < m; i++)
    new_values[i] = values[i];

  new_values[m] = new real [n];
  for (unsigned int i = 0; i < n; i++)
    new_values[m][i] = 0.0;
  
  m = m + 1;
  
  delete [] values;
  values = new_values;

  clearperm();
}
//-----------------------------------------------------------------------------
void DenseMatrix::addrow(const Vector &x)
{
  if ( x.size() != n)
    dolfin_error("Matrix dimensions don't match");
                                                                                                                                                            
  real** new_values = new real * [m+1];
                                                                                                                                                            
  for (unsigned int i = 0; i < m; i++)
    new_values[i] = values[i];
                                                                                                                                                            
  new_values[m] = new real[n];
  for (unsigned int i = 0; i < n; i++)
    new_values[m][i] = x(i);
                                                                                                                                                            
  m = m + 1;
                                                                                                                                                            
  delete [] values;
  values = new_values;

  clearperm();
}
//-----------------------------------------------------------------------------
void DenseMatrix::initrow(unsigned int i, unsigned int rowsize)
{
  dolfin_warning("Specifying number of non-zero elements on a row has no effect for a dense matrix.");
}
//-----------------------------------------------------------------------------
bool DenseMatrix::endrow(unsigned int i, unsigned int pos) const
{
  dolfin_warning("You probably don't want to use endrow() for a dense matrix.");
  return pos < n;
}
//-----------------------------------------------------------------------------
void DenseMatrix::settransp(const DenseMatrix& A)
{
  init(A.n, A.m);
                                                                                                                                                            
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      values[i][j] = A.values[j][i];
}
//-----------------------------------------------------------------------------
void DenseMatrix::settransp(const SparseMatrix& A)
{
  init(A.n, A.m);
                                                                                                                                                            
  unsigned int j;                                                                                                                                                            
  for (unsigned int i = 0; i < n; i++)
    for (unsigned int pos = 0; !A.endrow(i,pos); pos++)
      values[j][i] = A(i,j,pos);
}
//-----------------------------------------------------------------------------
void DenseMatrix::show() const 
{
  for (unsigned int i = 0; i < m; i++) {
    cout << "| ";
    for (unsigned int j = 0; j < n; j++){
      cout << (*this)(i,j) << " ";
    }
    cout << "|" << endl;
  }
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const DenseMatrix& A)
{
  unsigned int m = A.size(0);
  unsigned int n = A.size(1);
  
  stream << "[ Dense matrix of size " << m << " x " << n << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
void DenseMatrix::alloc(unsigned int m, unsigned int n)
{
  // Use with caution. Only for internal use.

  values = new real * [m];
  
  for (unsigned int i = 0; i < m; i++)
    values[i] = new real[n];
  
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      values[i][j] = 0.0;
 
  this->m = m;
  this->n = n;
}
//-----------------------------------------------------------------------------
real DenseMatrix::read(unsigned int i, unsigned int j) const
{
  return values[i][j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::write(unsigned int i, unsigned int j, real value)
{
  values[i][j] = value;
}
//-----------------------------------------------------------------------------
void DenseMatrix::add(unsigned int i, unsigned int j, real value)
{
  values[i][j] += value;
}
//-----------------------------------------------------------------------------
void DenseMatrix::sub(unsigned int i, unsigned int j, real value)
{
  values[i][j] -= value;
}
//-----------------------------------------------------------------------------
void DenseMatrix::mult(unsigned int i, unsigned int j, real value)
{
  values[i][j] *= value;
}
//-----------------------------------------------------------------------------
void DenseMatrix::div(unsigned int i, unsigned int j, real value)
{
  values[i][j] /= value;
}
//-----------------------------------------------------------------------------
real** DenseMatrix::getvalues()
{
  return values;
}
//-----------------------------------------------------------------------------
real** const DenseMatrix::getvalues() const
{
  return values;
}
//-----------------------------------------------------------------------------
void DenseMatrix::initperm()
{
  if ( !permutation )
    permutation = new unsigned int[m];
  
  for (unsigned int i = 0; i < m; i++)
    permutation[i] = i;
}
//-----------------------------------------------------------------------------
void DenseMatrix::clearperm()
{
  if ( permutation )
    delete [] permutation;
  permutation = 0;
}
//-----------------------------------------------------------------------------
unsigned int* DenseMatrix::getperm()
{
  return permutation;
}
//-----------------------------------------------------------------------------
unsigned int* const DenseMatrix::getperm() const
{
  return permutation;
}
//-----------------------------------------------------------------------------
