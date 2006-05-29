// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-04-03
// Last changed: 2006-05-15

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/DenseVector.h>
#include <dolfin/DenseMatrix.h>

// FIXME: Boost first

// These two files must be included due to a bug in Boost version < 1.33.
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>


using namespace dolfin;

//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix() : GenericMatrix(),
			     Variable("A", "a dense matrix"),
			     ublas_matrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(uint M, uint N) : GenericMatrix(),
					   Variable("A", "a dense matrix"),
					   ublas_matrix(M, N)
{
  // Clear matrix (not done by ublas)
  clear();
}
//-----------------------------------------------------------------------------
DenseMatrix::~DenseMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------

void DenseMatrix::init(uint M, uint N)
{
  if( this->size(0) == M && this->size(1) == N )
  {
    clear();
    return;
  }
  
  // Resize matrix (entries are not preserved)
  this->resize(M, N, false);

  // Clear matrix (not done by ublas)
  clear();
}
//-----------------------------------------------------------------------------
void DenseMatrix::init(uint M, uint N, uint nz)
{
  init(M, N);
}
//-----------------------------------------------------------------------------
dolfin::uint DenseMatrix::size(uint dim) const
{
  dolfin_assert( dim < 2 );
  return (dim == 0 ? this->size1() : this->size2());  
}
//-----------------------------------------------------------------------------
void DenseMatrix::set(const real block[], const int rows[], int m, const int cols[], int n)
{
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      (*this)(rows[i] , cols[j]) = block[i*n + j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::add(const real block[], const int rows[], int m, const int cols[], int n)
{
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      (*this)(rows[i] , cols[j]) += block[i*n + j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::solve(DenseVector& x, const DenseVector& b) const
{    
  // Make copy of matrix (factorisation is done in-place)
  DenseMatrix Atemp = *this;

  // Solve
  Atemp.solve_in_place(x, b);
}
//-----------------------------------------------------------------------------
void DenseMatrix::solve_in_place(DenseVector& x, const DenseVector& b)
{
  // This function does not check for singularity of the matrix
  uint M = this->size1();
  dolfin_assert(M == this->size2());
  dolfin_assert(M == b.size());
  
  // Initialise solution vector
  x = b;

  // Create permutation matrix
  ublas::permutation_matrix<std::size_t> pmatrix(M);

  // Factorise (with pivoting)
  ublas::lu_factorize(*this, pmatrix);
  
  // Back substitute 
  ublas::lu_substitute(*this, pmatrix, x);
}
//-----------------------------------------------------------------------------
void DenseMatrix::invert()
{
  // This function does not check for singularity of the matrix
  uint M = this->size1();
  dolfin_assert(M == this->size2());
  
  // Create permutation matrix
  ublas::permutation_matrix<std::size_t> pmatrix(M);

  // Set what will be the inverse inverse to identity matrix
  DenseMatrix inverse(M, M);
  inverse.assign(ublas::identity_matrix<real>(M));

  // Factorise 
  ublas::lu_factorize(*this, pmatrix);
  
  // Back substitute 
  ublas::lu_substitute(*this, pmatrix, inverse);

  *this = inverse;
}
//-----------------------------------------------------------------------------
void DenseMatrix::apply()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void DenseMatrix::zero()
{
  clear();
}
//-----------------------------------------------------------------------------
void DenseMatrix::ident(const int rows[], int m)
{
  uint n = this->size(1);
  for(int i=0; i < m; ++i)
    ublas::row(*this, rows[i]) = ublas::unit_vector<double> (n, rows[i]);
}
//-----------------------------------------------------------------------------
void DenseMatrix::mult(const DenseVector& x, DenseVector& Ax) const
{
  ublas::axpy_prod(*this, x, Ax, false);

//  axpy_prod() should be more efficient than this
//  Ax = prod(*this, x);
}
//-----------------------------------------------------------------------------
void DenseMatrix::disp(uint precision) const
{
  std::cout.precision(precision+1);
  
  std::cout << *this << std::endl;
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const DenseMatrix& A)
{
  // Check if matrix has been defined
  if ( A.size(0) == 0 || A.size(1) == 0 )
  {
    stream << "[ DenseMatrix matrix (empty) ]";
    return stream;
  }

  uint M = A.size(0);
  uint N = A.size(1);
  stream << "[ DenseMatrix matrix of size " << M << " x " << N << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
