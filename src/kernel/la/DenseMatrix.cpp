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
  if ( m <= 0 )
    dolfin_error("Number of rows must be positive.");

  if ( n <= 0 )
    dolfin_error("Number of columns must be positive.");
  
  values = 0;
  permutation = 0;
  
  init(m,n);
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(const DenseMatrix& A) : Variable("A", "A dense matrix")
{
  values = 0;
  permutation = 0;

  init(A.m, A.n);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      values[i][j] = A.values[i][j];
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(const Matrix& A) : Variable("A", "A dense matrix")
{
  values = 0;
  permutation = 0;

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
  if ( m <= 0 )
    dolfin_error("Number of rows must be positive.");

  if ( n <= 0 )
    dolfin_error("Number of columns must be positive.");
  
  if ( values ){
    for (int i = 0; i < m; i++)
      delete [] values[i];
    
    delete [] values;
    delete [] permutation;
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
  if ( values ) {
    for (int i = 0; i < m; i++)
      delete [] values[i];
    delete [] values;
    values = 0;
  }

  if ( permutation )
    delete [] permutation;
  permutation = 0;
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
void DenseMatrix::solve(Vector& x, const Vector& b)
{
  DirectSolver solver;
  solver.solve(*this, x, b);
}
//-----------------------------------------------------------------------------
void DenseMatrix::inverse(DenseMatrix& Ainv)
{
  DirectSolver solver;
  solver.inverse(*this, Ainv);
}
//-----------------------------------------------------------------------------
void DenseMatrix::hpsolve(Vector& x, const Vector& b) const
{
  DirectSolver solver;
  solver.hpsolve(*this, x, b);
}
//-----------------------------------------------------------------------------
void DenseMatrix::lu()
{
  DirectSolver solver;
  solver.lu(*this);
}
//-----------------------------------------------------------------------------
void DenseMatrix::solveLU(Vector& x, const Vector& b) const
{
  DirectSolver solver;
  solver.solveLU(*this, x, b);
}
//-----------------------------------------------------------------------------
void DenseMatrix::inverseLU(DenseMatrix& Ainv) const
{
  DirectSolver solver;
  solver.inverseLU(*this, Ainv);
}
//-----------------------------------------------------------------------------
void DenseMatrix::hpsolveLU(const DenseMatrix& LU,
			    Vector& x, const Vector& b) const
{
  DirectSolver solver;
  solver.hpsolveLU(LU, *this, x, b);
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
