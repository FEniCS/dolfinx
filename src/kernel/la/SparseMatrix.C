// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Display.hh>
#include "SparseMatrix.hh"
#include "DenseMatrix.hh"
#include "Vector.hh"

//-----------------------------------------------------------------------------
SparseMatrix::SparseMatrix()
{
  m    = 0;
  n    = 0;
  size = 0;

  locations = 0;
  columns   = 0;
  values    = 0;
}
//-----------------------------------------------------------------------------
SparseMatrix::SparseMatrix(int m, int n, int *ncols)
{
  locations = 0;
  columns   = 0;
  values    = 0;
  
  Resize(m,n,ncols);
}
//-----------------------------------------------------------------------------
SparseMatrix::~SparseMatrix()
{
  if ( locations )
	 delete [] locations;
  if ( columns )
	 delete [] columns;
  if ( values )
	 delete [] values;
}
//-----------------------------------------------------------------------------
void SparseMatrix::Resize(int m, int n, int *ncols)
{
  // Delete old data
  if ( locations )
	 delete locations;
  if ( columns )
	 delete columns;
  if ( values )
	 delete values;
  
  this->m = m;
  this->n = n;
  size = 0;
  
  // Compute the total size
  for (int i=0;i<m;i++){
	 if ( ncols[i] > n )
		display->InternalError("SparseMatrix::SparseMatrix()",
									  "Inconsistent initialisation: ncols[%d] = %d > %d = n.",
									  i,ncols[i],n);
	 size += ncols[i];
  }
  
  // Allocate memory
  locations = new int[m+1];
  columns   = new int[size];
  values    = new real[size];
  
  if ( !locations || !columns || !values )
	 display->Error("Unable to allocate memory for sparse matrix.");
  
  // Set the locations
  locations[0] = 0;
  for (int i=0;i<m;i++)
	 locations[i+1] = locations[i] + ncols[i];
  
  // Set the columns and the values (-1 means no value)
  int pos;
  for (int i=0;i<size;i++){
	 columns[i] = -1;
	 values[i]  = 0.0;
  }
  
}
//-----------------------------------------------------------------------------
void SparseMatrix::Reset()
{
  for (int i=0;i<size;i++){
	 values[i] = 0.0;
	 columns[i] = -1;
  }
}
//-----------------------------------------------------------------------------
void SparseMatrix::Set(int i, int j, int pos, real val)
{
  if ( (i<0) || (i>=m) || (j<0) || (j>=n) )
    display->InternalError("SparseMatrix::Set()","Illegal indices: (%d,%d).",i,j);
  
  int ncols = locations[i+1] - locations[i];
  
  if ( (pos<0) || (pos>=ncols) )
    display->InternalError("SparseMatrix::Set()","Illegal position: %d.",pos);
  
  int index = locations[i] + pos;
  
  columns[index] = j;
  values[index]  = val;
}
//-----------------------------------------------------------------------------
void SparseMatrix::Set(int i, int  j, real val)
{
  if ( (i<0) || (i>=m) || (j<0) || (j>=n) )
    display->InternalError("SparseMatrix::Set()","Illegal indices: (%d,%d).",i,j);

  int index;
  
  for (index=locations[i];index<locations[i+1];index++){

	 if ( columns[index] == j ){
		values[index] = val;
		return;
	 }
	 else if ( columns[index] == -1 ){
		columns[index] = j;
		values[index] = val;
		return;
	 }

  }

  display->InternalError("SparseMatrix::Set()","No empty position for element indices: (%d,%d).",i,j);
}
//-----------------------------------------------------------------------------
void SparseMatrix::Add(int i, int  j, real val)
{
  if ( (i<0) || (i>=m) || (j<0) || (j>=n) )
    display->InternalError("SparseMatrix::Add()","Illegal indices: (%d,%d).",i,j);

  int index;
  
  for (index=locations[i];index<locations[i+1];index++){

	 if ( columns[index] == j ){
		values[index] += val;
		return;
	 }
	 else if ( columns[index] == -1 ){
		columns[index] = j;
		values[index] += val;
		return;
	 }

  }
  
  display->InternalError("SparseMatrix::Add()","No empty position for element indices: (%d,%d).",i,j);
}
//-----------------------------------------------------------------------------
void SparseMatrix::Mult(SparseMatrix* B, SparseMatrix* AB)
{
  int mm = B->Size(0);
  int nn = B->Size(1);
  
  if ( n!=mm )
    display->InternalError("SparseMatrix::Mult()","Matrices not compatible.");

  display->InternalError("SparseMatrix::Mult()","Not implemented.");
}
//-----------------------------------------------------------------------------
void SparseMatrix::Mult(Vector* x, Vector* Ax)
{
  int nn = x->Size();
  int mm = Ax->Size();
  
  if ( (m!=mm) || (n!=nn) )
    display->InternalError("SparseMatrix::Mult()","Wrong dimensions for multiplication.");

  real sum;
  int pos0, pos1, j;

  for (int i=0;i<m;i++){

	 sum  = 0.0;
	 pos0 = locations[i];
	 pos1 = locations[i+1];

	 for (int pos=pos0;pos<pos1;pos++){
		j = columns[pos];
		if ( j == -1 )
		  break;
		sum += values[pos] * x->Get(j);
	 }
	 Ax->Set(i,sum);
	 
  }
  
}
//-----------------------------------------------------------------------------
real SparseMatrix::Mult(int i, Vector* x)
{
  int nn = x->Size();
  
  if ( n!=nn )
    display->InternalError("SparseMatrix::Mult()","Wrong dimensions for multiplication.");

  if ( (i<0) || (i>=m) )
    display->InternalError("SparseMatrix::Mult()","Illegal row index for multiplication.");

  real sum;
  int pos0, pos1, j;
  
  sum  = 0.0;
  pos0 = locations[i];
  pos1 = locations[i+1];
  
  for (int pos=pos0;pos<pos1;pos++){
	 j = columns[pos];
	 if ( j == -1 )
		break;
	 sum += values[pos] * x->Get(j);
  }

  return sum;
}
//-----------------------------------------------------------------------------
real SparseMatrix::Get(int i, int *j, int pos)
{
  // No checking of arguments to speed up the computation,.

  int index = locations[i] + pos;
  *j = columns[index];
  
  return ( values[index] );
}
//-----------------------------------------------------------------------------
real SparseMatrix::GetDiagonal(int i)
{
  if ( (i<0) || (i>=m) )
    display->InternalError("SparseMatrix::GetDiagonal()","Illegal row: %d",i);
  
  for (int pos=locations[i];pos<locations[i+1];pos++){
	 if ( columns[pos] == i )
		return values[pos];
	 else if ( columns[pos] == -1 )
		return 0.0;
  }

  return 0.0;
}
//-----------------------------------------------------------------------------
void SparseMatrix::CopyTo(DenseMatrix *A)
{
  int j;

  for (int i=0;i<m;i++)
	 for (int pos=locations[i];pos<locations[i+1];pos++){
		if ( (j=columns[pos]) == -1 )
		  break;
		A->Set(i,j,values[pos]);
	 }
  
}
//-----------------------------------------------------------------------------
real SparseMatrix::Norm()
{
  real max = 0.0;
  
  for (int i=0; i < size; i++)
    if ( fabs(values[i]) > max )
		max = fabs(values[i]);
  
  return ( max );
}
//-----------------------------------------------------------------------------
int SparseMatrix::GetRowLength(int i)
{
  if ( size == 0 )
	 return 0;
  
  if ( (i<0) || (i>=m) )
    display->InternalError("SparseMatrix::GetRowLength()","Illegal row index: %d.",i);

  return ( locations[i+1] - locations[i] );
}
//-----------------------------------------------------------------------------
SparseMatrix *SparseMatrix::GetCopy()
{
  int *ncols = new int[m];
  if ( !ncols )
	 display->Error("Unable to allocate memory for copy of sparse matrix.");

  for (int i=0;i<m;i++)
	 ncols[i] = locations[i+1] - locations[i];
  
  SparseMatrix *copy = new SparseMatrix(m,n,ncols);

  if ( !copy )
	 display->Error("Unable to allocate memory for copy of sparse matrix.");
  
  for (int i=0;i<m;i++)
	 for (int pos=0;pos<ncols[i];pos++)
		copy->Set(i,columns[locations[i]+pos],pos,values[locations[i]+pos]);
  
  delete ncols;
  
  return ( copy );
}
//-----------------------------------------------------------------------------
void SparseMatrix::Transpose()
{
  // Optimized w.r.t. extra memory requirements, not speed.

  // Allocate temporary memory
  int *tmp = new int[n];
  int *new_locations = new int[n+1];
  if ( !tmp || !new_locations )
	 display->Error("Unable to allocate temporary memory when computing tranpose.");

  // Compute new row lengths
  for (int i=0;i<n;i++)
	 tmp[i] = 0;
  for (int i=0;i<size;i++)
	 tmp[columns[i]] += 1;

  // Compute new locations
  new_locations[0] = 0;
  for (int i=0;i<n;i++)
	 new_locations[i+1] = new_locations[i] + tmp[i];

  // Allocate memory for new values
  real *new_values = new real[size];
  if ( !new_values )
	 display->Error("Unable to allocate temporary memory when computing transpose.");

  // Prepare column positions in new matrix
  for (int i=0;i<n;i++)
	 tmp[i] = 0;

  // Set the new values
  int new_row;
  for (int i=0;i<m;i++)
	 for (int pos=locations[i];pos<locations[i+1];pos++){
		new_row = columns[pos];
		new_values[new_locations[new_row] + tmp[new_row]++] = values[pos];
	 }

  // Set the new values
  delete values;
  values = new_values;

  // Allocate memory for new column indices
  int *new_columns = new int[size];
  if ( !new_columns )
	 display->Error("Unable to allocate temporary memory when computing transpose.");
  
  // Prepare column positions in new matrix
  for (int i=0;i<n;i++)
	 tmp[i] = 0;
  
  // Compute new columns
  for (int i=0;i<m;i++)
	 for (int pos=locations[i];pos<locations[i+1];pos++){
		new_row = columns[pos];
		new_columns[new_locations[new_row] + tmp[new_row]++] = i;
	 }
  
  // Set the new columns
  delete columns;
  columns = new_columns;

  // Delete temporary memory
  delete tmp;
  delete new_columns;

  // Set the dimensions;
  new_row = n;
  n = m;
  m = new_row;
}
//-----------------------------------------------------------------------------
void SparseMatrix::Display()
{
  display->Message(0,"Sparse matrix of size %d x %d with %d nonzero elements.",
						 m,n,size);
}
//-----------------------------------------------------------------------------
void SparseMatrix::DisplayAll()
{
  Display();

  for (int i=0;i<m;i++){
	 for (int pos=locations[i];pos<locations[i+1];pos++)
		printf("%f ",values[pos]);
	 printf("\n");
  }
  
}
//-----------------------------------------------------------------------------
void SparseMatrix::DropZeros(real tol)
{
  if ( tol < 0.0 )
	 display->InternalError("SparseMatrix::DropZeros()","Tolerance cannot be negative: tol = %f.\n",tol);

  // Mark values that should be removed
  for (int i=0;i<m;i++)
	 for (int pos=locations[i];pos<locations[i+1];pos++)
		if ( fabs(values[pos]) <= tol )
		  columns[pos] = -1;

  // Remove values
  
  display->InternalError("SparseMatrix::DropZeros()","Not finished yet...");

}
//-----------------------------------------------------------------------------
void SparseMatrix::SetRowIdentity(int i)
{
  if ( (i<0) || (i>=m) )
	 display->InternalError("SparseMatrix::SetRowIdentity()",
									"Illegal row index: %d",i);

  int pos0 = locations[i];
  int pos1 = locations[i+1];
  columns[pos0] = i;
  values[pos0]  = 1.0;
  for (int pos=pos0+1;pos<pos1;pos++){
	 columns[pos] = -1;
	 values[pos]  = 0.0;
  }

}
//-----------------------------------------------------------------------------
void SparseMatrix::SetRowIdentity(int *rows, int no_rows)
{
  if ( no_rows > m  )
	 display->InternalError("SparseMatrix::SetRowIdentity()",
									"Too many rows: %d > m = %d.",no_rows,m);
  
  for (int i=0;i<no_rows;i++)
	 SetRowIdentity(rows[i]);
}
//-----------------------------------------------------------------------------
void SparseMatrix::ScaleRow(int i, real a)
{
  for (int pos=locations[i];pos<locations[i+1];pos++)
	 values[pos] *= a;
}
//-----------------------------------------------------------------------------
int SparseMatrix::Size(int i)
{
  if (i==0){
    return m;
  } else if (i==1){
    return n;
  } else{
    display->Error("Wrong argument, have to be 0 or 1.");
  }    
}
//-----------------------------------------------------------------------------
