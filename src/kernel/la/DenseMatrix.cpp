// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/DirectSolver.h>
#include <dolfin/Matrix.h>
#include <dolfin/DenseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(int m, int n) : Variable("A", "A dense matrix")
{
  if ( m <= 0 || n <= 0 )
    dolfin_error("Illegal dimensions.");
  
  values = 0;
  permutation = 0;
  
  init(m,n);
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(Matrix& A) : Variable("A", "A dense matrix")
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
  if ( m <= 0 || n <= 0 )
    dolfin_error("DenseMatrix::init(): Illegal dimensions.");
  
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
  for (int i = 0; i < m; i++)
    delete [] values[i];
  
  delete [] values;
  delete [] permutation;
}
//-----------------------------------------------------------------------------
real& DenseMatrix::operator() (int i, int j)
{
  return values[i][j];
}
//-----------------------------------------------------------------------------
real DenseMatrix::operator() (int i, int j) const
{
  return values[i][j];
}
//-----------------------------------------------------------------------------
int DenseMatrix::size(int dim) const
{
  if ( dim == 0 )
    return m;
  else
    return n;
}
//-----------------------------------------------------------------------------
void DenseMatrix::solve(Vector& x, Vector& b)
{
  DirectSolver solver;

  solver.solve(*this, x, b);
}
//-----------------------------------------------------------------------------
void DenseMatrix::lu()
{
  DirectSolver solver;

  solver.lu(*this);
}
//-----------------------------------------------------------------------------
void DenseMatrix::lusolve(Vector& x, Vector& b)
{
  DirectSolver solver;

  solver.lusolve(*this, x, b);
}
//-----------------------------------------------------------------------------
