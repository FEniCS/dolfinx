// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-04-03
// Last changed: 2006-06-26

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/DenseVector.h>
#include <dolfin/DenseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix() : GenericMatrix(),
			     Variable("A", "a dense matrix"),
			     ublas_dense_matrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
DenseMatrix::DenseMatrix(const uint M, const uint N) : GenericMatrix(),
					   Variable("A", "a dense matrix"),
					   ublas_dense_matrix(M, N)
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

void DenseMatrix::init(const uint M, const uint N)
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
void DenseMatrix::init(const uint M, const uint N, const uint nz)
{
  init(M, N);
}
//-----------------------------------------------------------------------------
dolfin::uint DenseMatrix::size(const uint dim) const
{
  dolfin_assert( dim < 2 );
  return (dim == 0 ? this->size1() : this->size2());  
}
//-----------------------------------------------------------------------------
void DenseMatrix::set(const real block[], const int rows[], int m, const int cols[], const int n)
{
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      (*this)(rows[i] , cols[j]) = block[i*n + j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::add(const real block[], const int rows[], int m, const int cols[], const int n)
{
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      (*this)(rows[i] , cols[j]) += block[i*n + j];
}
//-----------------------------------------------------------------------------
void DenseMatrix::lump(DenseVector& m) const
{
  ublas::scalar_vector<double> one(size(1), 1.0);
  ublas::axpy_prod(*this, one, m, true);
}
//-----------------------------------------------------------------------------
void DenseMatrix::solve(DenseVector& x, const DenseVector& b) const
{    
  // Make copy of matrix (factorisation is done in-place)
  DenseMatrix Atemp = *this;

  // Solve
  Atemp.solveInPlace(x, b);
}
//-----------------------------------------------------------------------------
void DenseMatrix::solveInPlace(DenseVector& x, const DenseVector& b)
{
  // This function does not check for singularity of the matrix
  const uint M = this->size1();
  dolfin_assert(M == this->size2());
  dolfin_assert(M == b.size());
  
   if( x.size() != M )
    x.init(M);

  // Initialise solution vector
  x.assign(b);

  // Create permutation matrix
  ublas::permutation_matrix<std::size_t> pmatrix(M);

  // Factorise (with pivoting)
  uint singular = ublas::lu_factorize(*this, pmatrix);
  if( singular > 0)
    dolfin_error1("Singularity detected in uBlas matrix factorization on line %u.", singular-1); 
  
  // Back substitute 
  ublas::lu_substitute(*this, pmatrix, x);
}
//-----------------------------------------------------------------------------
void DenseMatrix::invert()
{
  // This function does not check for singularity of the matrix
  const uint M = this->size1();
  dolfin_assert(M == this->size2());
  
  // Create permutation matrix
  ublas::permutation_matrix<std::size_t> pmatrix(M);

  // Set what will be the inverse inverse to identity matrix
  DenseMatrix inverse(M, M);
  inverse.assign(ublas::identity_matrix<real>(M));

  // Factorise (with pivoting)
  uint singular = ublas::lu_factorize(*this, pmatrix);
  if( singular > 0)
    dolfin_error1("Singularity detected in uBlas matrix factorization on line %u.", singular-1); 
  
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
void DenseMatrix::ident(const int rows[], const int m)
{
  const uint n = this->size(1);
  for(int i=0; i < m; ++i)
    ublas::row(*this, rows[i]) = ublas::unit_vector<double> (n, rows[i]);
}
//-----------------------------------------------------------------------------
void DenseMatrix::mult(const DenseVector& x, DenseVector& Ax) const
{
  ublas::axpy_prod(*this, x, Ax, true);
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
