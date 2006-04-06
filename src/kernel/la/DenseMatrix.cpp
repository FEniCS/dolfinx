// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-04-03
// Last changed: 2006-04-06

#include <dolfin/dolfin_log.h>
#include <dolfin/DenseMatrix.h>

// These two files must be included due to a bug in Boost version < 1.33.
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>

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
dolfin::uint DenseMatrix::size(uint dim) const
{
  dolfin_assert( dim < 2 );
  return (dim == 0 ? this->size1() : this->size2());  
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
