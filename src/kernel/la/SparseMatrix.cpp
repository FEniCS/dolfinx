// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Georgios Foufas 2002, 2003
// Modified by Erik Svensson, 2003.
// Modified by Karin Kraft, 2004.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/Vector.h>
#include <dolfin/DenseMatrix.h>
#include <dolfin/SparseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SparseMatrix::SparseMatrix():
  rowsizes(0),
  columns(0),
  values(0),
  allocsize(1)
{
  m = 0;
  n = 0;
  
}
//-----------------------------------------------------------------------------
SparseMatrix::SparseMatrix(unsigned int m, unsigned int n) :
  rowsizes(0),
  columns(0),
  values(0),
  allocsize(1)
{
  this->m = 0;
  this->n = 0;
  
  init(m,n);
}
//-----------------------------------------------------------------------------
SparseMatrix::SparseMatrix(const SparseMatrix& A) :
  rowsizes(0),
  columns(0),
  values(0),
  allocsize(1)
{
  this->m = A.m;
  this->n = A.n;
  
  if ( m == 0 )
    return;
  
  // We cannot use init since we want to allocate memory to exactly match
  // the structure of the given SparseMatrix A.

  rowsizes = new unsigned int [m];
  columns = new int * [m];
  values = new real * [m];

  for (unsigned int i = 0; i < m; i++) {
    unsigned int rs = (rowsizes[i] = A.rowsizes[i]);
    columns[i] = new int [rs];
    values[i] = new real [rs];
    
    for (unsigned int pos = 0; pos < rs; pos++) {
      columns[i][pos] = A.columns[i][pos];
      values[i][pos] = A.values[i][pos];
    }
  }
}
//-----------------------------------------------------------------------------
SparseMatrix::SparseMatrix(const DenseMatrix& A):
  rowsizes(0),
  columns(0),
  values(0),
  allocsize(1)
{
  this->m = A.m;
  this->n = A.n;
  
  if ( m == 0 )
    return;
  
  // We cannot use init since we want to allocate memory to exactly match
  // the structure (based on the number of non-zero elements) of the
  // given DenseMatrix A.

  rowsizes = new unsigned int [m];
  columns = new int * [m];
  values = new real * [m];
  
  for (unsigned int i = 0; i < m; i++) {
    
    // Count number of nonzero elements on row
    unsigned int rs = 0;
    for (unsigned int j = 0; j < n; j++)
      if ( fabs(A.values[i][j]) > DOLFIN_EPS )
	rs++;

    rowsizes[i] = rs;
    columns[i] = new int [rs];
    values[i] = new real [rs];
    
    unsigned int pos = 0;
    for (unsigned int j = 0; j < n; j++)
      if ( fabs(A.values[i][j]) > DOLFIN_EPS ) {
	columns[i][pos] = j;
	values[i][pos] = A.values[i][j];
	pos++;
      }
  }
}
//-----------------------------------------------------------------------------
SparseMatrix::~SparseMatrix()
{
  clear();
}
//-----------------------------------------------------------------------------
void SparseMatrix::init(unsigned int m, unsigned int n)
{
  if ( m < 0 )
    dolfin_error("Number of rows must be positive.");
  
  if ( n < 0 )
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
void SparseMatrix::clear()
{
  if ( rowsizes )
    delete [] rowsizes;
  rowsizes = 0;
  
  if ( columns ){
    for (unsigned int i = 0; i < m; i++)
      if ( columns[i] )
	delete [] columns[i];
    delete [] columns;
  }
  columns = 0;
  
  if ( values ){
    for (unsigned int i = 0; i < m; i++)
      if ( values[i] )
	delete [] values[i];
    delete [] values;
  }
  values = 0;
  
  allocsize = 1;
  
  m = 0;
  n = 0;
}
//-----------------------------------------------------------------------------
unsigned int SparseMatrix::size(unsigned int dim) const
{
  if ( dim == 0 )
    return m;
  else if ( dim == 1 )
    return n;
  
  dolfin_warning1("Illegal matrix dimension: dim = %d.", dim);
  return 0;
}
//-----------------------------------------------------------------------------
unsigned int SparseMatrix::size() const
{  
  unsigned int sum = 0;
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; pos < rowsizes[i]; pos++)
      if ( columns[i][pos] != -1 )
	sum += 1;
  
  return sum;
}
//-----------------------------------------------------------------------------
unsigned int SparseMatrix::rowsize(unsigned int i) const
{
  return rowsizes[i];
}
//-----------------------------------------------------------------------------
unsigned int SparseMatrix::bytes() const
{
  unsigned int bytes = 0;
  
  bytes += sizeof(SparseMatrix);
  bytes += m * sizeof(int);
  bytes += m * sizeof(int *);
  bytes += m * sizeof(real *);
  for (unsigned int i = 0; i < m; i++)
	 bytes += rowsizes[i] * ( sizeof(int) + sizeof(real) );

  return bytes;
} 
//-----------------------------------------------------------------------------
real SparseMatrix::operator()(unsigned int i, unsigned int j) const
{
  return read(i,j);
}
//-----------------------------------------------------------------------------
real* SparseMatrix::operator[](unsigned int i)
{
  dolfin_error("Using dense quick-access operator for sparse matrix.");
  return 0;
}
//-----------------------------------------------------------------------------
real SparseMatrix::operator()(unsigned int i, unsigned int& j, unsigned int pos) const
{
  j = columns[i][pos];
  
  return values[i][pos];
}
//-----------------------------------------------------------------------------
void SparseMatrix::operator=(real a)
{
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; pos < rowsizes[i]; pos++)
      values[i][pos] = a;
}
//-----------------------------------------------------------------------------
void SparseMatrix::operator=(const DenseMatrix& A)
{
  dolfin_error("Assignment from dense matrix to sparse matrix. Not recommended.");
}
//-----------------------------------------------------------------------------
void SparseMatrix::operator=(const SparseMatrix& A)
{
  init(A.m, A.n);
  
  for (unsigned int i = 0; i < m; i++) {

    initrow(i, A.rowsizes[i]);   

    for (unsigned int pos = 0; pos < rowsizes[i]; pos++) {
      columns[i][pos] = A.columns[i][pos];
      values[i][pos] = A.values[i][pos];
    }

  }
}
//-----------------------------------------------------------------------------
void SparseMatrix::operator+= (const DenseMatrix& A)
{
  dolfin_error("Adding dense matrix to sparse matrix. Not recommended.");
}
//-----------------------------------------------------------------------------
void SparseMatrix::operator+= (const SparseMatrix& A)
{
  if ( A.m != m || A.n != n )
    dolfin_error("Matrix dimensions don't match.");
  
  int j = 0;
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; pos < A.rowsizes[i]; pos++) {
      if ( (j = A.columns[i][pos]) == -1 )
	break;
      add(i, j, A.values[i][pos]);
    }
}
//-----------------------------------------------------------------------------
void SparseMatrix::operator-= (const DenseMatrix& A)
{
  dolfin_error("Subtracting dense matrix from sparse matrix. Not recommended.");
}
//-----------------------------------------------------------------------------
void SparseMatrix::operator-= (const SparseMatrix& A)
{
  if ( A.m != m || A.n != n )
    dolfin_error("Matrix dimensions don't match.");
  
  int j = 0;
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; pos < A.rowsizes[i]; pos++) {
      if ( (j = A.columns[i][pos]) == -1 )
	break;
      sub(i, j, A.values[i][pos]);
    }
}
//-----------------------------------------------------------------------------
void SparseMatrix::operator*= (real a)
{
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < rowsizes[i]; j++)
      values[i][j] *= a;
}
//-----------------------------------------------------------------------------
real SparseMatrix::norm() const
{
  real max = 0.0;
  real val;
  
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; pos < rowsizes[i]; pos++)
      if ( (val = fabs(values[i][pos])) > max )
	max = val;
  
  return max;
}
//-----------------------------------------------------------------------------
real SparseMatrix::mult(const Vector& x, unsigned int i) const
{
  if ( n != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  real sum = 0.0;
  for (unsigned int pos = 0; pos < rowsizes[i] && columns[i][pos] != -1; pos++)
    sum += values[i][pos] * x(columns[i][pos]);

  return sum;
}
//-----------------------------------------------------------------------------
void SparseMatrix::mult(const Vector& x, Vector& Ax) const
{ 
 if ( n != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  Ax.init(m);
  
  for (unsigned int i = 0; i < m; i++)
    Ax(i) = mult(x, i);
}
//-----------------------------------------------------------------------------
void SparseMatrix::multt(const Vector& x, Vector& Ax) const
{
  if ( m != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  Ax.init(n);
  Ax = 0.0;
                                                                                                                                                         
  for (unsigned int i = 0; i < m; i++)
    for (unsigned int pos = 0; pos < rowsizes[i] && columns[i][pos] != -1; pos++)
      Ax(columns[i][pos]) += values[i][pos] * x(i);
}
//-----------------------------------------------------------------------------
real SparseMatrix::multrow(const Vector& x, unsigned int i) const
{
  return mult(x,i);
}
//-----------------------------------------------------------------------------
real SparseMatrix::multcol(const Vector& x, unsigned int j) const
{
  dolfin_error("Computing scalar product with column of sparse matrix. Not recommended.");
  return 0.0;
}
//-----------------------------------------------------------------------------
void SparseMatrix::resize()
{
  unsigned int oldsize = 0;
  unsigned int newsize = 0;
  allocsize = 0;
  
  for (unsigned int i = 0; i < m; i++) {
    
    // Count number of used elements
    unsigned int rowsize = 0;
    for (unsigned int pos = 0; pos < rowsizes[i]; pos++)
      if ( columns[i][pos] != -1 )
	rowsize++;
    
    // Keep track of number of cleared elements
    oldsize += rowsizes[i];
    newsize += rowsize;
    
    // Keep track of maximum row length
    if ( rowsize > allocsize )
      allocsize = rowsize;
    
    // Allocate new row
    int*  cols = new int[rowsize];
    real* vals = new real[rowsize];
    
    // Copy old elements
    unsigned int newpos = 0;
    for (unsigned int pos = 0; pos < rowsizes[i]; pos++)
      if ( columns[i][pos] != -1 ) {
	cols[newpos] = columns[i][pos];
	vals[newpos] = values[i][pos];
	newpos++;
      }
    
    // Change to new elements
    delete [] columns[i];
    delete [] values[i];
    columns[i]  = cols;
    values[i]   = vals;
    rowsizes[i] = rowsize;
    
  }
  
  // Write a message
  cout << "Cleared " << (oldsize - newsize) << " unused elements." << endl;
}
//-----------------------------------------------------------------------------
void SparseMatrix::ident(unsigned int i)
{
  if ( columns[i] )
    delete [] columns[i];
  if ( values[i] )
    delete [] values[i];
  
  columns[i] = new int[1];
  values[i] = new real[1];
  
  columns[i][0] = i;
  values[i][0] = 1.0;

  rowsizes[i] = 1;
}
//-----------------------------------------------------------------------------
void SparseMatrix::lump(Vector& a) const
{
  a.init(m);

  for (unsigned int i = 0; i < m; i++)
  {
    a(i) = 0.0;
    for (unsigned int pos = 0; pos < rowsizes[i]; pos++)
      a(i) += values[i][pos];
  }
}
//-----------------------------------------------------------------------------
void SparseMatrix::addrow()
{
  real** new_values = new real * [m+1];
  int** new_columns = new int * [m+1];
  unsigned int* new_rowsizes = new unsigned int [m+1];
                                                                                                                                                            
  for (unsigned int i = 0; i < m; i++) {
    new_values[i] = values[i];
    new_columns[i] = columns[i];
    new_rowsizes[i] = rowsizes[i];
  }
                                                                                                                                                            
  new_values[m] = new real [1];
  new_columns[m] = new int [1];
  new_rowsizes[m] = 1;
  new_values[m][0] = 0.0;
  new_columns[m][0] = -1;
                                                                                                                                                            
  m = m + 1;
                                                                                                                                                            
  delete [] values;
  delete [] columns;
  delete [] rowsizes;
  values = new_values;
  columns = new_columns;
  rowsizes = new_rowsizes;
}
//-----------------------------------------------------------------------------
void SparseMatrix::addrow(const Vector& x)
{
  if ( n != x.size() )
    dolfin_error("Matrix dimensions don't match.");

  real **new_values = new real * [m+1];
  int **new_columns = new int * [m+1];
  unsigned int *new_rowsizes = new unsigned int [m+1];
                                                                                                                                                            
  for (unsigned int i = 0; i < m; i++) {
    new_values[i] = values[i];
    new_columns[i] = columns[i];
    new_rowsizes[i] = rowsizes[i];
  }
                                                                                                                                                            
  unsigned int nonzero = 0;
  for (unsigned int i = 0; i < x.size(); i++)
    if (fabs(x(i)) > DOLFIN_EPS)
      nonzero++;

  new_values[m] = new real [nonzero];
  new_columns[m] = new int [nonzero];
  new_rowsizes[m] = nonzero;
                     
  unsigned int pos = 0;
  for (unsigned int i = 0; i < x.size(); i++)
    if (fabs(x(i)) > DOLFIN_EPS) {
      new_values[m][pos] = x(i);
      new_columns[m][pos] = i;
      pos++;
    }
  
  if (nonzero > allocsize)
    allocsize = nonzero;
                                                                                                                                                            
  m = m + 1;
                                                                                                                                                            
  delete [] values;
  delete [] columns;
  delete [] rowsizes;
  values = new_values;
  columns = new_columns;
  rowsizes = new_rowsizes;
}
//-----------------------------------------------------------------------------
void SparseMatrix::initrow(unsigned int i, unsigned int rowsize)
{
  if ( columns[i] ){
    delete [] columns[i];
    columns[i] = new int[rowsize];
  }
  
  if ( values[i] ){
    delete [] values[i];
    values[i] = new real[rowsize];
  }
  
  for (unsigned int pos = 0; pos < rowsize; pos++){
    columns[i][pos] = -1;
    values[i][pos] = 0.0;
  }
  
  rowsizes[i] = rowsize;
}
//-----------------------------------------------------------------------------
bool SparseMatrix::endrow(unsigned int i, unsigned int pos) const
{
  // FIXME: Replace with an iterator

  if ( pos >= rowsizes[i] )
    return true;
  
  return columns[i][pos] == -1;
}
//-----------------------------------------------------------------------------
void SparseMatrix::settransp(const SparseMatrix& A)
{
  clear();
                                                                                                                                                            
  this->m = A.n;
  this->n = A.m;
                                                                                                                                                            
  // Allocate memory
  rowsizes = new unsigned int [m];
  columns  = new int * [m];
  values   = new real * [m];

  // Current position in row
  unsigned int* rowpos = new unsigned int [m];

  // Reset row sizes
  for (unsigned int i = 0; i < m; i++) {
    rowsizes[i] = 0;
    rowpos[i] = 0;
  }

  // Sum up row sizes
  for (unsigned int i = 0; i < n; i++ )
    for (unsigned int pos = 0; pos < A.rowsizes[i]; pos++)
      ++rowsizes[A.columns[i][pos]];
  
  // Allocate memory
  allocsize = 0;
  for (unsigned int i = 0; i < m; i++) {
    columns[i] = new int [rowsizes[i]];
    values[i] = new real [rowsizes[i]];
    allocsize = (rowsizes[i] > allocsize ) ? rowsizes[i] : allocsize;
  }

  // Set values
  for (unsigned int i = 0; i < n; i++)
    for (unsigned int pos = 0; pos < A.rowsizes[i]; pos++) {
      int k = A.columns[i][pos];
      values[k][rowpos[k]] = A.values[i][pos];
      columns[k][rowpos[k]] = i;
      ++rowpos[k];
    }
                                                                                                                                                            
  delete [] rowpos;
}
//-----------------------------------------------------------------------------
void SparseMatrix::settransp(const DenseMatrix& A)
{
  clear();
                                                                                                                                                            
  this->m = A.n;
  this->n = A.m;

  // Allocate memory
  rowsizes = new unsigned int [m];
  columns = new int * [m];
  values = new real * [m];

  // Current position in row
  unsigned int* rowpos = new unsigned int [m];
          
  // Reset row sizes
  for (unsigned int i = 0; i < m; i++) {
    rowsizes[i] = 0;
    rowpos[i] = 0;
  }

  // Sum up row sizes
  for (unsigned int i = 0; i < n; i++ )
    for (unsigned int j = 0; j < m; j++)
      if (fabs(A.values[i][j]) > DOLFIN_EPS)
        ++rowsizes[j];

  // Allocate memory
  allocsize = 0;
  for (unsigned int i = 0; i < m; i++) {
    columns[i] = new int[rowsizes[i]];
    values[i] = new real[rowsizes[i]];
    allocsize = (rowsizes[i] > allocsize) ? rowsizes[i] : allocsize;
  }

  // Set values
  for (unsigned int i = 0; i < n; i++)
    for (unsigned int j = 0; j < m; j++)
      if (fabs(A.values[i][j]) > DOLFIN_EPS) {
        values[j][rowpos[j]] = A.values[i][j];
        columns[j][rowpos[j]] = i;
        ++rowpos[j];
      }
  
  delete [] rowpos;
}
//-----------------------------------------------------------------------------
void SparseMatrix::show() const
{
  cout << "Sparse matrix" << endl;


  for (unsigned int i = 0; i < m; i++) {
    cout << "| ";
    for (unsigned int j = 0; j < n; j++){
      cout << (*this)(i,j) << " ";
    }
    cout << "|" << endl;
  }
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream, const SparseMatrix& A)
{
  unsigned int size = A.size();
  unsigned int bytes = A.bytes();
  
  unsigned int m = A.size(0);
  unsigned int n = A.size(1);
  
  stream << "[ Sparse matrix of size " << m << " x " << n << " with " << size;
  stream << " nonzero entries, approx ";
  
  if ( bytes > 1024*1024 )
    stream << bytes/1024 << " Mb. ]";
  else if ( bytes > 1024 )
    stream << bytes/1024 << " kb. ]";
  else
    stream << bytes << " bytes ]";

  return stream;
}
//-----------------------------------------------------------------------------
void SparseMatrix::alloc(unsigned int m, unsigned int n)
{
  // Use with caution. Only for internal use.
  
  rowsizes = new unsigned int [m];
  columns = new int * [m];
  values = new real * [m];

  for (unsigned int i = 0; i < m; i++) {
    columns[i] = new int [1];
    values[i] = new real [1];
    rowsizes[i] = 1;
    columns[i][0] = -1;
    values[i][0] = 0.0;
  }
  
  this->m = m;
  this->n = n;

  allocsize = 1;
}
//-----------------------------------------------------------------------------
real SparseMatrix::read(unsigned int i, unsigned int j) const
{
  if ( i < 0 || i >= m )
    dolfin_error1("Illegal row index: i = %d.", i);

  if ( j < 0 || j >= n )
    dolfin_error1("Illegal column index: j = %d.", j);
  
  for (unsigned int pos = 0; pos < rowsizes[i]; pos++)
    if ( columns[i][pos] == static_cast<int>(j) )
      return values[i][pos];
  
  return 0.0;
}
//-----------------------------------------------------------------------------
void SparseMatrix::write(unsigned int i, unsigned int j, real value)
{
  dolfin_assert(i < m);
  dolfin_assert(j < n);

  // Find position (i,j)
  unsigned int pos = 0;
  for (; pos < rowsizes[i]; pos++){
    // Put element in already existing position
    if ( columns[i][pos] == static_cast<int>(j) ) {
      values[i][pos] = value;
      return;
    }
    // Put element in an unused position
    else if ( columns[i][pos] == -1 ) {
      columns[i][pos] = j;
      values[i][pos] = value;
      return;
    }
  }
  
  // Couldn't find an empty position, so resize and try again
  resizeRow(i, rowsizes[i] + 1);
  
  // Insert new element (requires same ordering as before resizeRow())
  columns[i][pos] = j;
  values[i][pos] = value;
}
//-----------------------------------------------------------------------------
void SparseMatrix::add(unsigned int i, unsigned int j, real value)
{
  // Use first empty position
  unsigned int pos = 0;
  for (; pos < rowsizes[i]; pos++){
    if ( columns[i][pos] == static_cast<int>(j) ) {
      values[i][pos] += value;
      return;
    }
    else if ( columns[i][pos] == -1 ) {
      columns[i][pos] = j;
      values[i][pos] = value;
      return;
    }
  }
  
  // Couldn't find an empty position, so resize and try again
  resizeRow(i, rowsizes[i] + 1);

  // Insert new element (requires same ordering as before resizeRow())
  columns[i][pos] = j;
  values[i][pos] = value;
}
//-----------------------------------------------------------------------------
void SparseMatrix::sub(unsigned int i, unsigned int j, real value)
{
  add(i, j, -value);
}
//-----------------------------------------------------------------------------
void SparseMatrix::mult(unsigned int i, unsigned int j, real value)
{
  unsigned int pos = 0;
  for (; pos < rowsizes[i]; pos++){
    if ( columns[i][pos] == static_cast<int>(j) ) {
      values[i][pos] *= value;
      return;
    }
  }
}
//-----------------------------------------------------------------------------
void SparseMatrix::div(unsigned int i, unsigned int j, real value)
{
  mult(i, j, 1.0/value);
}
//-----------------------------------------------------------------------------
real** SparseMatrix::getvalues()
{
  return values;
}
//-----------------------------------------------------------------------------
real** const SparseMatrix::getvalues() const
{
  return values;
}
//-----------------------------------------------------------------------------
void SparseMatrix::initperm()
{
  dolfin_error("Trying to initialize permutation for a sparse matrix.");
}
//-----------------------------------------------------------------------------
void SparseMatrix::clearperm()
{
  dolfin_error("Trying to clear permutation for a sparse matrix.");
}
//-----------------------------------------------------------------------------
unsigned int* SparseMatrix::getperm()
{
  dolfin_error("Trying to access permutation for a sparse matrix.");

  return 0;
}
//-----------------------------------------------------------------------------
unsigned int* const SparseMatrix::getperm() const
{
  dolfin_error("Trying to access permutation for a sparse matrix.");

  return 0;
}
//----------------------------------------------------------------------------- 
void SparseMatrix::resizeRow(unsigned int i, unsigned int rowsize)
{
  // Allocate at least allocsize
  if ( allocsize > rowsize )
    rowsize = allocsize;
  else if ( rowsize > allocsize )
    allocsize = rowsize;
  
  // Allocate new lists
  int* cols = new int[rowsize];
  real* vals = new real[rowsize];
  
  // Copy values
  for (unsigned int pos = 0; pos < rowsizes[i] || pos < rowsize; pos++) {
    if ( pos < rowsizes[i] ) {
      cols[pos] = columns[i][pos];
      vals[pos] = values[i][pos];
    }
    else {
      cols[pos] = -1;
      vals[pos] = 0.0;
    }
  }
  
  // Delete old values and use the new lists
  delete [] values[i];
  delete [] columns[i];
  
  columns[i] = cols;
  values[i] = vals;
  rowsizes[i] = rowsize;
}
//-----------------------------------------------------------------------------
