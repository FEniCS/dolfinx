// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Contributions by: Georgios Foufas 2002, 2003

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
Matrix::Matrix()
{
  m = 0;
  n = 0;
  
  rowsizes = 0;
  columns  = 0;
  values   = 0;

  allocsize = 1;

  rename("A", "A sparse matrix");
}
//-----------------------------------------------------------------------------
Matrix::Matrix(int m, int n)
{
  this->m = m;
  this->n = n;
  
  rowsizes = 0;
  columns  = 0;
  values   = 0;

  allocsize = 1;
  
  init(m,n);

  rename("A", "A sparse matrix");
}
//-----------------------------------------------------------------------------
Matrix::~Matrix()
{
  clear();
}
//-----------------------------------------------------------------------------
void Matrix::operator= (const Matrix& A)
{
  init(A.m, A.n);
  
  for (int i = 0; i < m; i++) {
    initRow(i,A.rowsizes[i]);
    
    for (int pos = 0; pos < rowsizes[i]; pos++) {
      columns[i][pos] = A.columns[i][pos];
      values[i][pos] = A.values[i][pos];
    }
  }
}
//-----------------------------------------------------------------------------
void Matrix::operator+= (const Matrix& A)
{
  if ( A.m != m || A.n != n ) {
	// FIXME: Use logging system
	cout << "Matrix::operator= (): Matrices not compatible." << endl;
	exit(1);
  }

  int j = 0;
  for (int i = 0; i < m; i++)
    for (int pos = 0; pos < A.rowsizes[i]; pos++) {
      if ( (j = A.columns[i][pos]) == -1 )
		  break;
		addtoElement(i, j, A.values[i][pos]);
    }
}
//-----------------------------------------------------------------------------
void Matrix::operator-= (const Matrix& A)
{
  if ( A.m != m || A.n != n ) {
	// FIXME: Use logging system
	cout << "Matrix::operator= (): Matrices not compatible." << endl;
	exit(1);
  }

  int j = 0;
  for (int i = 0; i < m; i++)
    for (int pos = 0; pos < A.rowsizes[i]; pos++) {
      if ( (j = A.columns[i][pos]) == -1 )
                  break;
                addtoElement(i, j, -A.values[i][pos] );
    }
}
//-----------------------------------------------------------------------------
void Matrix::operator*= (real a)
{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < rowsizes[i]; j++)
      values[i][j] *= a;
}
//-----------------------------------------------------------------------------
void Matrix::init(int m, int n)
{
  // Check arguments

  // FIXME: Use logging system
  if ( m < 0 ) {
    cout << "Matrix::init(): Number of rows must be positive." << endl;
	 exit(1);
  }
  if ( n < 0 ) {
    cout << "Matrix::init(): Number of columns must be positive." << endl;
	 exit(1);
  }
  
  // Delete old data
  clear();
  
  this->m = m;
  this->n = n;
  
  // Allocate memory
  rowsizes = new int[m];
  columns  = new (int *)[m];
  values   = new (real *)[m];
  
  if ( !columns || !values ) {
	 cout << "Unable to allocate memory for sparse matrix." << endl;
	 exit(1);
  }
	 
  // Create an empty matrix
  for (int i = 0; i < m; i++){
	 columns[i] = new int[1];
	 values[i] = new real[1];
	 
	 rowsizes[i] = 1;
	 columns[i][0] = -1;
	 values[i][0] = 0.0;
  }

  allocsize = 1;
}
//-----------------------------------------------------------------------------
void Matrix::resize()
{
  int oldsize = 0;
  int newsize = 0;
  allocsize = 0;
  
  for (int i = 0; i < m; i++) {

	 // Count number of used elements
	 int rowsize = 0;
	 for (int pos = 0; pos < rowsizes[i]; pos++)
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
	 int newpos = 0;
	 for (int pos = 0; pos < rowsizes[i]; pos++)
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
  cout << "Clearing " << (oldsize - newsize) << " unused elements." << endl;
}
//-----------------------------------------------------------------------------
void Matrix::clear()
{
  if ( rowsizes )
	 delete [] rowsizes;
  rowsizes = 0;
	 
	 if ( columns ){
	 for (int i = 0; i < m; i++)
	 if ( columns[i] )
	 delete [] columns[i];
	 delete [] columns;
  }
  columns = 0;
  
  if ( values ){
	 for (int i = 0; i < m; i++)
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
int Matrix::size(int dim) const
{
  if ( dim == 0 )
	 return m;
  else if ( dim == 1 )
	 return n;

  cout << "Matrix::size(): Illegal dimension" << endl;
  return 0;
}
//-----------------------------------------------------------------------------
int Matrix::size() const
{  
  int size = 0;
  for (int i = 0; i < m; i++)
	 for (int pos = 0; pos < rowsizes[i]; pos++)
		if ( columns[i][pos] != -1 )
		  size += 1;

  return size;
}
//-----------------------------------------------------------------------------
int Matrix::bytes() const
{
  int bytes = 0;
  
  bytes += sizeof(Matrix);
  bytes += m * sizeof(int);
  bytes += m * sizeof(int *);
  bytes += m * sizeof(real *);
  for (int i = 0; i < m; i++)
	 bytes += rowsizes[i] * ( sizeof(int) + sizeof(real) );

  return bytes;
} 
//-----------------------------------------------------------------------------
void Matrix::initRow(int i, int rowsize)
{
  if ( i < 0 || i >= m )
    cout << "Matrix::initRow(): Illegal row index" << endl;
  if ( rowsize < 0 )
    cout << "Matrix::initRow(): Illegal row size" << endl;
  
  if ( columns[i] ){
	 delete [] columns[i];
	 columns[i] = new int[rowsize];
  }
  
  if ( values[i] ){
	 delete [] values[i];
	 values[i] = new real[rowsize];
  }

  for (int pos = 0; pos < rowsize; pos++){
	 columns[i][pos] = -1;
	 values[i][pos] = 0.0;
  }
  
  rowsizes[i] = rowsize;
}
//-----------------------------------------------------------------------------
void Matrix::resizeRow(int i, int rowsize)
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
  for (int pos = 0; pos < rowsizes[i] || pos < rowsize; pos++) {
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
int Matrix::rowSize(int i) const
{
  if ( i < 0 || i >= m )
    cout << "Matrix::rowSize(): Illegal row index" << endl;

  return rowsizes[i];
}
//-----------------------------------------------------------------------------
bool Matrix::endrow(int i, int pos) const
{
  if ( pos >= rowsizes[i] )
	 return true;

  return columns[i][pos] == -1;
}
//-----------------------------------------------------------------------------
real Matrix::operator()(int i, int *j, int pos) const
{
  if ( i < 0 || i >= m )
	 cout << "Matrix::operator (): Illegal row index" << endl;

  if ( pos >= rowsizes[i] )
	 cout << "Matrix::operator (): Illegal position" << endl;

  *j = columns[i][pos];
  
  return values[i][pos];
}
//-----------------------------------------------------------------------------
Matrix::Reference Matrix::operator()(int i, int j)
{
  return Reference(*this, i, j);
}
//-----------------------------------------------------------------------------
real Matrix::operator()(int i, int j) const
{
  return readElement(i, j);
}
//-----------------------------------------------------------------------------
real Matrix::norm()
{
  real max = 0.0;
  real val;
  
  for (int i = 0; i < m; i++)
	 for (int pos = 0; pos < rowsizes[i]; pos++)
		if ( (val = fabs(values[i][pos])) > max )
		  max = val;
  
  return ( max );
}
//-----------------------------------------------------------------------------
void Matrix::ident(int i)
{
  if ( i < 0 || i >= m )
    cout << "Matrix::setRowIdentity(): Illegal row index" << endl;

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
real Matrix::mult(Vector &x, int i)
{
  if ( i < 0 || i >= m )
    cout << "Matrix::mult(): Illegal row index" << endl;

  real sum = 0.0;

  for (int pos = 0; pos < rowsizes[i] && columns[i][pos] != -1; pos++)
	 sum += values[i][pos] * x(columns[i][pos]);

  return sum;
}
//-----------------------------------------------------------------------------
void Matrix::mult(Vector &x, Vector &Ax)
{
  if ( x.size() != n || Ax.size() != n )
	 cout << "Matrix::mult(): Matrix dimensions don't match." << endl;
  
  for (int i = 0; i < m; i++)
	 Ax(i) = mult(x,i);
}
//-----------------------------------------------------------------------------
void Matrix::show()
{
  for (int i = 0; i < m; i++) {
	 if ( i == 0 )
		cout << "| ";
	 else
		cout << "| ";
	 for (int j = 0; j < n; j++){
		cout << (*this)(i,j) << " ";
	 }
	 cout << "|" << endl;
  }
}
//-----------------------------------------------------------------------------
real Matrix::readElement(int i, int j) const
{
  if ( i < 0 || i >= m || j < 0 || j >= n )
    cout << "Matrix::readElement(): Illegal indices" << endl;

  for (int pos = 0; pos < rowsizes[i]; pos++)
	 if ( columns[i][pos] == j )
		return values[i][pos];

  return 0.0;
}
//-----------------------------------------------------------------------------
void Matrix::writeElement(int i, int j, real value)
{
  if ( i < 0 || i >= m || j < 0 || j >= n )
    cout << "Matrix::operator(): Illegal indices" << endl;

  // Find position (i,j)
  int pos = 0;
  for (; pos < rowsizes[i]; pos++){
	 // Put element in already existing position
	 if ( columns[i][pos] == j ){
		values[i][pos] = value;
		return;
	 }
	 // Put element in an unused position
	 else if ( columns[i][pos] == -1 ){
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
void Matrix::addtoElement(int i, int j, real value)
{
  if ( i < 0 || i >= m || j < 0 || j >= n )
    cout << "Matrix::operator(): Illegal indices" << endl;

  // Use first empty position
  int pos = 0;
  for (; pos < rowsizes[i]; pos++){
	 if ( columns[i][pos] == j ){
		values[i][pos] += value;
		return;
	 }
	 else if ( columns[i][pos] == -1 ){
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
// Additional operators
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Matrix& A)
{
  int size = A.size();
  int bytes = A.bytes();

  int m = A.size(0);
  int n = A.size(1);
  
  stream << "[ Sparse matrix of size " << m << " x " << n << " with " << size;
  stream << " nonzero entries, approx ";
  
  if ( bytes > 1024*1024 )
	 stream << bytes/1024 << " Mb. ]";
  else if ( bytes > 1024 )
	 stream << bytes/1024 << " kb. ]";
  else
	 stream << bytes << " bytes. ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
