// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-03
// Last changed: 2006-04-07

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/DenseMatrix.h>

// These two files must be included due to a bug in Boost version < 1.33.
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


using namespace dolfin;

//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix() : boost::numeric::ublas::matrix<real>(), 
    Variable("A", "a dense matrix")
{
  //Do nothing
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(uint i, uint j) : 
    boost::numeric::ublas::matrix<double>(i, j), Variable("A", "a dense matrix")
{
  //Do nothing
}
//-----------------------------------------------------------------------------
DenseMatrix::~DenseMatrix()
{
  //Do nothing
}
//-----------------------------------------------------------------------------
void DenseMatrix::init(uint N, uint M)
{
  if( this->size(0) == N && this->size(1) == M )
    return;
  
  this->resize(N, M, false);
}
//-----------------------------------------------------------------------------
void DenseMatrix::init(uint N, uint M, uint nz)
{
  init(N, M);
}
//-----------------------------------------------------------------------------
dolfin::uint DenseMatrix::size(uint dim) const
{
  dolfin_assert( dim < 2 );
  return (dim == 0 ? this->size1() : this->size2());  
}
//-----------------------------------------------------------------------------
void DenseMatrix::add(const real block[], const int rows[], int m, const int cols[], int n)
{
  for(int i = 0; i < m; ++i)    // loop over rows
    for(int j = 0; j < n; ++j)  // loop over columns
      (*this)( rows[i] , cols[j] ) += *(block + i*n + j);
}
//-----------------------------------------------------------------------------
void DenseMatrix::invert()
{
  // This function does not check for singularity of the matrix

  uint m = this->size1();
  uint n = this->size2();
  dolfin_assert(m == n);
  
  // Create permutation matrix
  boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix(m); 

  // Set what will be the inverse inverse to identity matrix
  DenseMatrix inverse(m,m);
  inverse.assign(boost::numeric::ublas::identity_matrix<real>(m));

  // Factorise 
  boost::numeric::ublas::lu_factorize(*this, pmatrix);
  
  // Back substitute 
  boost::numeric::ublas::lu_substitute(*this, pmatrix, inverse);

  *this = inverse;  
}
//-----------------------------------------------------------------------------
void DenseMatrix::disp(uint precision) const
{
  std::cout.precision(precision);
  
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

  uint m = A.size(0);
  uint n = A.size(1);
  stream << "[ DenseMatrix matrix of size " << m << " x " << n << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
