// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Andy R. Terrel, 2005.
//
// First added:  2005-01-17
// Last changed: 2005-10-24

#include <iostream>

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/VirtualMatrix.h>

using namespace dolfin;

// Mult function
namespace dolfin
{
 
  int usermult(Mat A, Vec x, Vec y)
  {
    void* ctx = 0;
    MatShellGetContext(A, &ctx);
    Vector xx(x), yy(y);
    ((VirtualMatrix*) ctx)->mult(xx, yy);
    return 0;
  }

}

//-----------------------------------------------------------------------------
VirtualMatrix::VirtualMatrix() : A(0)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
VirtualMatrix::VirtualMatrix(const Vector& x, const Vector& y) : A(0)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  init(x, y);
}
//-----------------------------------------------------------------------------
VirtualMatrix::~VirtualMatrix()
{
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void VirtualMatrix::init(const Vector& x, const Vector& y)
{
  // Get size and local size of given vector
  int m(0), n(0), M(0), N(0);
  VecGetLocalSize(y.vec(), &m);
  VecGetLocalSize(x.vec(), &n);
  VecGetSize(y.vec(), &M);
  VecGetSize(x.vec(), &N);
  
  // Free previously allocated memory if necessary
  if ( A )
  {
    // Get size and local size of existing matrix
    int mm(0), nn(0), MM(0), NN(0);
    MatGetLocalSize(A, &mm, &nn);
    MatGetSize(A, &MM, &NN);

    if ( mm == m && nn == n && MM == M && NN == N )
      return;
    else
    {
      MatDestroy(A);
    }
  }
  
  MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, (void*) this, &A);
  MatShellSetOperation(A, MATOP_MULT, (void (*)()) usermult);
}
//-----------------------------------------------------------------------------
void VirtualMatrix::init(int M, int N)
{
  // Put here to set up arbitrary Shell of global size M,N.
  // Analagous to the matrix being on one processor. 

  // Free previously allocated memory if necessary
  if ( A )
    {
      // Get size and local size of existing matrix                                                            
      int MM(0), NN(0);
      MatGetSize(A, &MM, &NN);

      if ( MM == M && NN == N )
	return;
      else
	MatDestroy(A);
    }

  MatCreateShell(PETSC_COMM_WORLD, M, N, M, N, (void*) this, &A);
  MatShellSetOperation(A, MATOP_MULT, (void (*)()) usermult);
}
//-----------------------------------------------------------------------------
dolfin::uint VirtualMatrix::size(uint dim) const
{
  int M = 0;
  int N = 0;
  MatGetSize(A, &M, &N);
  dolfin_assert(M >= 0);
  dolfin_assert(N >= 0);

  return (dim == 0 ? static_cast<uint>(M) : static_cast<uint>(N));
}
//-----------------------------------------------------------------------------
Mat VirtualMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
void VirtualMatrix::disp(bool sparse, int precision) const
{
  // Since we don't really have the matrix, we create the matrix by
  // performing multiplication with unit vectors. Used only for debugging.
  
  uint M = size(0);
  uint N = size(1);
  Vector x(N), y(M);
  Matrix A(M, N);
  
  x = 0.0;
  for (unsigned int j = 0; j < N; j++)
  {
    x(j) = 1.0;
    mult(x, y);
    for (unsigned int i = 0; i < M; i++)
    {
      const real value = y(i);
      if ( fabs(value) > DOLFIN_EPS )
	A(i, j) = value;
    }
    x(j) = 0.0;
  }

  A.disp(sparse, precision);
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const VirtualMatrix& A)
{
  MatType type = 0;
  MatGetType(A.mat(), &type);
  int m = A.size(0);
  int n = A.size(1);
  stream << "[ PETSc matrix (type " << type << ") of size "
	 << m << " x " << n << " ]";

  return stream;
}
//-----------------------------------------------------------------------------
