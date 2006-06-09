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
#include <dolfin/uBlasSparseMatrix.h>

// These two files must be included due to a bug in Boost version < 1.33.
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>


using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasSparseMatrix::uBlasSparseMatrix() : GenericMatrix(),
			     Variable("A", "a sparse matrix"),
			     ublas_sparse_matrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasSparseMatrix::uBlasSparseMatrix(uint M, uint N) : GenericMatrix(),
					   Variable("A", "a sparse matrix"),
					   ublas_sparse_matrix(M, N)
{
  // Clear matrix (not done by ublas)
  clear();
}
//-----------------------------------------------------------------------------
uBlasSparseMatrix::~uBlasSparseMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::init(uint M, uint N)
{
  if( this->size(0) == M && this->size(1) == N )
    return;
  
  // Resize matrix (entries are not preserved)
  this->resize(M, N, false);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::init(uint M, uint N, uint nz)
{
  init(M, N);

  uint total_nz = nz*(this->size(0));
  reserve(total_nz);
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasSparseMatrix::size(uint dim) const
{
  dolfin_assert( dim < 2 );
  return (dim == 0 ? this->size1() : this->size2());  
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::set(const real block[], const int rows[], int m, const int cols[], int n)
{
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      (*this)(rows[i] , cols[j]) = block[i*n + j];
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::add(const real block[], const int rows[], int m, const int cols[], int n)
{
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      (*this)(rows[i] , cols[j]) += block[i*n + j];
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::getRow(const uint i, int& ncols, Array<int>& columns, 
    Array<real>& values) const
{
  // Reference to matrix row (through away const-ness and trust uBlas)
  ublas::matrix_row<uBlasSparseMatrix> row(*(const_cast<uBlasSparseMatrix*>(this)), i);

  // Iterator of components of row
  ublas::matrix_row<uBlasSparseMatrix>::iterator component;

  // Insert values into Arrays
  columns.clear();
  values.clear();
  for (component=row.begin(); component != row.end(); ++component) 
  {
    columns.push_back( component.index() );
    values.push_back( *component );
  }
  ncols = columns.size();
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::lump(DenseVector& m) const
{
  ublas::scalar_vector<double> one(size(1), 1.0);
  ublas::axpy_prod(*this, one, m, true);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::solve(DenseVector& x, const DenseVector& b) const
{    
  // Make copy of matrix (factorisation is done in-place)
  uBlasSparseMatrix Atemp = *this;

  // Solve
  Atemp.solveInPlace(x, b);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::solveInPlace(DenseVector& x, const DenseVector& b)
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
void uBlasSparseMatrix::apply()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::zero()
{
  // Clear destroys non-zero structure of the matrix 
  clear();

  // Set all non-zero values to zero without detroying non-zero pattern
//  (*this) *= 0.0;
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::ident(const int rows[], int m)
{
  uint n = this->size(1);
  for(int i=0; i < m; ++i)
    ublas::row(*this, rows[i]) = ublas::unit_vector<double> (n, rows[i]);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::mult(const DenseVector& x, DenseVector& Ax) const
{
  ublas::axpy_prod(*this, x, Ax, true);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::disp(uint precision) const
{
  std::cout.precision(precision+1);
  std::cout << *this << std::endl;
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const uBlasSparseMatrix& A)
{
  // Check if matrix has been defined
  if ( A.size(0) == 0 || A.size(1) == 0 )
  {
    stream << "[ uBlasSparseMatrix matrix (empty) ]";
    return stream;
  }

  uint M = A.size(0);
  uint N = A.size(1);
  stream << "[ uBlasSparseMatrix matrix of size " << M << " x " << N << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
