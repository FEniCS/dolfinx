// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Display.hh>
#include "DenseMatrix.hh"
#include <dolfin/Vector.h>
#include <dolfin/SparseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
SparseMatrix::SparseMatrix()
{
  m = 0;
  n = 0;
  
  rowsizes = 0;
  columns  = 0;
  values   = 0;
}
//-----------------------------------------------------------------------------
SparseMatrix::SparseMatrix(int m, int n)
{
  m = 0;
  n = 0;
  
  rowsizes = 0;
  columns  = 0;
  values   = 0;
  
  resize(m,n);
}
//-----------------------------------------------------------------------------
SparseMatrix::~SparseMatrix()
{
  clear();
}
//-----------------------------------------------------------------------------
void SparseMatrix::resize(int m, int n)
{
  // Check arguments
  if ( m < 0 )
    display->InternalError("SparseMatrix::resize()","Illegal number of rows: %d.",m);
  if ( n < 0 )
    display->InternalError("SparseMatrix::resize()","Illegal number of columns: %d.",n);
  
  // Delete old data
  clear();
  
  this->m = m;
  this->n = n;
  
  // Allocate memory
  rowsizes = new int[m];
  columns  = new (int *)[m];
  values   = new (real *)[m];
  
  if ( !columns || !values )
	 display->Error("Unable to allocate memory for sparse matrix.");
  
  // Create an empty matrix
  for (int i = 0; i < m; i++){
	 columns[i] = new int[1];
	 values[i] = new real[1];
	 
	 rowsizes[i] = 1;
	 columns[i][0] = -1;
	 values[i][0] = 0.0;
  }
  
}
//-----------------------------------------------------------------------------
void SparseMatrix::clear()
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
	 delete values;
  }
  values = 0;

  m = 0;
  n = 0;
}
//-----------------------------------------------------------------------------
int SparseMatrix::size(int dim)
{
  if ( dim == 0 )
	 return m;
  else if ( dim == 1 )
	 return n;

  display->InternalError("SparseMatrix::size()","Illegal dimension: %d.",dim);
}
//-----------------------------------------------------------------------------
int SparseMatrix::size()
{  
  int size = 0;
  for (int i = 0; i < m; i++)
	 for (int pos = 0; pos < rowsizes[i]; pos++)
		if ( columns[i][pos] != -1 )
		  size += 1;

  return size;
}
//-----------------------------------------------------------------------------
int SparseMatrix::bytes()
{
  int bytes = 0;
  
  bytes += sizeof(SparseMatrix);
  bytes += m * sizeof(int);
  bytes += m * sizeof(int *);
  bytes += m * sizeof(real *);
  for (int i = 0; i < m; i++)
	 bytes += rowsizes[i] * ( sizeof(int) + sizeof(real) );

  return bytes;
} 
//-----------------------------------------------------------------------------
void SparseMatrix::setRowSize(int i, int rowsize)
{
  if ( i < 0 || i >= m )
    display->InternalError("SparseMatrix::setRowSize()","Illegal row index: %d.",i);
  if ( rowsize < 0 )
    display->InternalError("SparseMatrix::setRowSize()","Illegal row size: %d.",rowsize);

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
int SparseMatrix::rowSize(int i)
{
  if ( i < 0 || i >= m )
    display->InternalError("SparseMatrix::rowSize()","Illegal row index: %d.",i);

  return rowsizes[i];
}
//-----------------------------------------------------------------------------
real SparseMatrix::operator()(int i, int *j, int pos) const
{
  if ( i < 0 || i >= m )
	 display->InternalError("SparseMatrix::operator ()",
									"Illegal row index: %d",i);

  if ( pos >= rowsizes[i] )
	 display->InternalError("SparseMatrix::operator ()",
									"Illegal position: %d",pos);

  *j = columns[i][pos];
  
  return values[i][pos];
}
//-----------------------------------------------------------------------------
SparseMatrix::Reference SparseMatrix::operator()(int i, int j)
{
 return Reference(this,i,j);
}
//-----------------------------------------------------------------------------
const real SparseMatrix::operator()(int i, int j) const
{
  return readElement(i,j);
}
//-----------------------------------------------------------------------------
real SparseMatrix::norm()
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
void SparseMatrix::setRowIdentity(int i)
{
  if ( i < 0 || i >= m )
    display->InternalError("SparseMatrix::setRowIdentity()","Illegal row index: %d.",i);

  if ( columns[i] )
	 delete columns[i];
  if ( values[i] )
	 delete [] values[i];

  columns[i] = new int[1];
  values[i] = new real[1];

  columns[i][0] = i;
  values[i][0] = 1.0;

  rowsizes[i] = 1;
}
//-----------------------------------------------------------------------------
real SparseMatrix::mult(Vector &x, int i)
{
  if ( i < 0 || i >= m )
    display->InternalError("SparseMatrix::mult()","Illegal row index: %d.",i);

  real sum = 0.0;

  for (int pos = 0; pos < rowsizes[i]; pos++)
	 sum += values[i][pos] * x(columns[i][pos]);

  return sum;
}
//-----------------------------------------------------------------------------
void SparseMatrix::mult(Vector &x, Vector &Ax)
{
  if ( x.size() != n || Ax.size() != n )
	 display->InternalError("SparseMatrix::mult()","Matrix dimensions don't match.");
  
  real sum;
  for (int i = 0; i < m; i++)
	 Ax(i) = mult(x,i);
}
//-----------------------------------------------------------------------------
void SparseMatrix::show()
{
  for (int i = 0; i < n; i++){
	 if ( i == 0 )
		cout << "A = | ";
	 else
		cout << "    | ";
	 for (int j = 0; j < n; j++){
		cout << (*this)(i,j) << " ";
	 }
	 cout << "|" << endl;
  }
}
//-----------------------------------------------------------------------------
real SparseMatrix::readElement(int i, int j) const
{
  if ( i < 0 || i >= m || j < 0 || j >= n )
    display->InternalError("SparseMatrix::readElement()","Illegal indices: (%d,%d).",i,j);

  for (int pos = 0; pos < rowsizes[i]; pos++)
	 if ( columns[i][pos] == j )
		return values[i][pos];

  return 0.0;
}
//-----------------------------------------------------------------------------
void SparseMatrix::writeElement(int i, int j, real value)
{
  if ( i < 0 || i >= m || j < 0 || j >= n )
    display->InternalError("SparseMatrix::operator()","Illegal indices: (%d,%d).",i,j);

  // Use first empty position
  for (int pos = 0; pos < rowsizes[i]; pos++){
	 if ( columns[i][pos] == j ){
		values[i][pos] = value;
		return;
	 }
	 else if ( columns[i][pos] == -1 ){
		columns[i][pos] = j;
		values[i][pos] = value;
		return;
	 }
  }
  
  display->InternalError("SparseMatrix::writeElement()","Row %d is full.",i);
}
//-----------------------------------------------------------------------------

namespace dolfin {

  //---------------------------------------------------------------------------
  ostream& operator << (ostream& output, SparseMatrix& sparseMatrix)
  {
	 int size = sparseMatrix.size();
	 int bytes = sparseMatrix.bytes();
	 	 
	 output << "[ Sparse matrix with " << size;
	 output << " nonzero entries, approx ";

	 if ( bytes > 1024*1024 )
		output << bytes/1024 << " Mb.]";
	 else if ( bytes > 1024 )
		output << bytes/1024 << " kb.]";
	 else
		output << bytes << " bytes.]";
	 
	 return output;
  }
  //---------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
