#include <stdio.h>

#include <dolfin/SparseMatrix.h>
#include <dolfin/DenseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(int m, int n)
{
  if ( (m<=0) || (n<=0) ) {
	 cout << "DenseMatrix::DenseMatrix(): Illegal dimensions" << endl;
	 exit(1);
  }
  
  values = 0;
  permutation = 0;
  
  init(m,n);
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(SparseMatrix &A)
{
  init(A.size(0), A.size(1));

  int j;
  real value;
  
  for (int i = 0; i < m; i++)
	 for (int pos = 0; pos < A.rowSize(i); pos++){
		value = A(i,&j,pos);
		values[i][j] = value;
	 }

}
//-----------------------------------------------------------------------------
void DenseMatrix::init(int m, int n)
{
  if ( m <= 0 || n <= 0 ) {
	 cout << "DenseMatrix::init(): Illegal dimensions" << endl;
	 exit(1);
  }
 
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
void DenseMatrix::set(int i, int j, real value)
{
  if ( (i < 0) || (j < 0) || (i >= m) || (j >= n) ) {
	 cout << "DenseMatrix::Set(): Indices out of range." << endl;
	 exit(1);
  }
	 
  values[i][j] = value;
}
//-----------------------------------------------------------------------------
int DenseMatrix::size(int dim)
{
  if ( dim == 0 )
	 return m;
  else
	 return n;
}
//-----------------------------------------------------------------------------
real DenseMatrix::get(int i, int j)
{
  if ( (i < 0) || (j < 0) || (i >= m) || (j >= n) ) {
	 cout << "DenseMatrix::Get(): Indices out of range." << endl;
	 exit(1);
  }

  return ( values[i][j] );
}

//-----------------------------------------------------------------------------
void DenseMatrix::DisplayAll()
{
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
