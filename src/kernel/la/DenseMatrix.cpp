#include <stdio.h>

#include "DenseMatrix.h"
#include <dolfin/SparseMatrix.h>
#include <dolfin/Display.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(int m, int n)
{
 if ( (m<=0) || (n<=0) )
	display->InternalError("DenseMatrix::DenseMatrix()",
								  "Illegal dimensions %d x %d.",m,n);

 values = 0;
 permutation = 0;
 
 resize(m,n);
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(SparseMatrix &A)
{
  resize(A.size(0),A.size(1));

  int j;
  real value;
  
  for (int i = 0; i < m; i++)
	 for (int pos = 0; pos < A.rowSize(i); pos++){
		value = A(i,&j,pos);
		values[i][j] = value;
	 }

}
//-----------------------------------------------------------------------------
void DenseMatrix::resize(int m, int n)
{
 if ( m <= 0 || n <= 0 )
	display->InternalError("DenseMatrix::Resize()",
								  "Illegal dimensions %d x %d.",m,n);
 
 if ( values ){
	for (int i=0;i<m;i++)
	  delete values[i];
	
	delete values;
	delete permutation;
 }

 this->m = m;
 this->n = n;
 
 values = new (real *)[m];
 for (int i=0;i<m;i++)
	values[i] = new real[n];
 
 for (int i=0;i<m;i++)
	for (int j=0;j<n;j++)
	  values[i][j] = 0.0;
 
 permutation = new int[m]; // for LU factorization
 for (int i=0;i<m;i++)
	permutation[i] = i;  
}
//-----------------------------------------------------------------------------
DenseMatrix::~DenseMatrix()
{
  for (int i=0;i<m;i++)
	 delete values[i];

  delete values;
  delete permutation;
}
//-----------------------------------------------------------------------------
void DenseMatrix::Set(int i, int j, real value)
{
  if ( (i < 0) || (j < 0) || (i >= m) || (j >= n) )
	 display->InternalError("DenseMatrix::Set()",
									"Indices (%d,%d) out of range.",i,j);

  values[i][j] = value;
}
//-----------------------------------------------------------------------------
int DenseMatrix::Size(int dim)
{
  if ( dim == 0 )
	 return m;
  else
	 return n;
}
//-----------------------------------------------------------------------------
real DenseMatrix::Get(int i, int j)
{
  if ( (i < 0) || (j < 0) || (i >= m) || (j >= n) )
	 display->InternalError("DenseMatrix::Get()",
									"Indices (%d,%d) out of range.",i,j);

  return ( values[i][j] );
}

//-----------------------------------------------------------------------------
void DenseMatrix::Display()
{
  display->Message(0,"Dense matrix of size %d x %d with %d elements.",
						 m,n,m*n);
}
//-----------------------------------------------------------------------------
void DenseMatrix::DisplayAll()
{
  Display();
  
  for (int i=0;i<m;i++){
	 for (int j=0;j<n;j++)
		printf("%f ",values[i][j]);
	 printf("\n");
  }
  
}
//-----------------------------------------------------------------------------
void DenseMatrix::DisplayRow(int i)
{
  printf("Row %d (%d): ",i,i+1);
  for (int j=0;j<n;j++)
	 printf("%f ",values[i][j]);
  printf("\n");
}
//-----------------------------------------------------------------------------
void DenseMatrix::Write(const char *filename)
{
  FILE *fp = fopen(filename,"w");

  if ( !fp )
	 display->Error("Unable to save matrix to file \"%s\".",filename);


  for (int i=0;i<m;i++){
	 for (int j=0;j<n;j++){
		fprintf(fp,"%1.16e",values[i][j]);
		if ( j < (n-1) )
		  fprintf(fp," ");
	 }
	 fprintf(fp,"\n");
  }

  fclose(fp);
}
//-----------------------------------------------------------------------------
