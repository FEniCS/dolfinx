// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-04-03
// Last changed: 2006-06-30

#include <iostream>
#include <dolfin/dolfin_log.h>
#include <dolfin/DenseVector.h>
#include <dolfin/uBlasSparseMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasSparseMatrix::uBlasSparseMatrix()
  : Variable("A", "a sparse matrix"),
    GenericMatrix(),
    uBlasKrylovMatrix(),
    ublas_sparse_matrix(), assembled(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasSparseMatrix::uBlasSparseMatrix(uint M, uint N)
  : Variable("A", "a sparse matrix"),
    GenericMatrix(),
    uBlasKrylovMatrix(),
    ublas_sparse_matrix(M, N), assembled(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasSparseMatrix::~uBlasSparseMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::init(const uint M, const uint N)
{
  // Resize matrix
  if( size(0) != M || size(1) != N )
    resize(M, N, false);
  
  // Resize assembly matrix
  if(Assembly_matrix.size1() != M && Assembly_matrix.size2() != N )
    Assembly_matrix.resize(M, N, false);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::init(const uint M, const uint N, const uint nz)
{
  init(M, N);

  // Reserve space for non-zeroes
  const uint total_nz = nz*size(0);
  reserve(total_nz);

  // This is not yet supported by the uBlas matrix type being used for assembly
//  Assembly_matrix.reserve(total_nz);  
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasSparseMatrix::size(const uint dim) const
{
  dolfin_assert( dim < 2 );
  return (dim == 0 ? size1() : size2());  
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::set(const real block[], const int rows[], int m, 
                            const int cols[], const int n)
{
  if( assembled )
  {
    Assembly_matrix.assign(*this);
    assembled = false; 
  }

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
        Assembly_matrix(rows[i] , cols[j]) = block[i*n + j];
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::add(const real block[], const int rows[], int m, 
                            const int cols[], const int n)
{
  if( assembled )
  {
    Assembly_matrix.assign(*this);
    assembled = false; 
  }

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      Assembly_matrix(rows[i] , cols[j]) += block[i*n + j];
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::getRow(const uint i, int& ncols, Array<int>& columns, 
    Array<real>& values) const
{
  if( !assembled )
    dolfin_error("Sparse matrix has not been assembled. Did you forget to call A.apply()?"); 

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
  if( !assembled )
    dolfin_error("Sparse matrix has not been assembled. Did you forget to call A.apply()?"); 

  ublas::scalar_vector<double> one(size(1), 1.0);
  ublas::axpy_prod(*this, one, m, true);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::solve(DenseVector& x, const DenseVector& b) const
{    
  if( !assembled )
    dolfin_error("Sparse matrix has not been assembled. Did you forget to call A.apply()?"); 

  // Make copy of matrix and vector (factorisation is done in-place)
  uBlasSparseMatrix Atemp = *this;

  // Solve
  Atemp.solveInPlace(x, b);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::solveInPlace(DenseVector& x, const DenseVector& b)
{
  if( !assembled )
    dolfin_error("Sparse matrix has not been assembled. Did you forget to call A.apply()?"); 

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
void uBlasSparseMatrix::apply()
{
  // Assign temporary assembly matrix to the sparse matrix
  if( !assembled )
  {
    // Assign temporary assembly matrix to the matrix
    this->assign(Assembly_matrix);
    assembled = true;
  } 

  // Free memory
  Assembly_matrix.resize(0,0, false);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::zero()
{
  if( !assembled )
    dolfin_error("Sparse matrix has not been assembled. Did you forget to call A.apply()?"); 

  // Clear destroys non-zero structure of the matrix 
  clear();

  // Set all non-zero values to zero without detroying non-zero pattern
//  (*this) *= 0.0;
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::ident(const int rows[], const int m)
{
  if( !assembled )
    dolfin_error("Sparse matrix has not been assembled. Did you forget to call A.apply()?"); 

  const uint n = this->size(1);
  for(int i=0; i < m; ++i)
    ublas::row(*this, rows[i]) = ublas::unit_vector<double> (n, rows[i]);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::mult(const DenseVector& x, DenseVector& Ax) const
{
  if( !assembled )
    dolfin_error("Sparse matrix has not been assembled. Did you forget to call A.apply()?"); 

  ublas::axpy_prod(*this, x, Ax, true);
}
//-----------------------------------------------------------------------------
void uBlasSparseMatrix::disp(const uint precision) const
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
