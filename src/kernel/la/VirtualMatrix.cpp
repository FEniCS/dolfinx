// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewVector.h>
#include <dolfin/VirtualMatrix.h>

using namespace dolfin;

// Mult function
namespace dolfin
{
  
  int usermult(Mat A, Vec x, Vec y)
  {
    void* ctx;
    MatShellGetContext(A, &ctx);
    ((VirtualMatrix*) ctx)->mult(x, y);
    return 0;
  }

}

//-----------------------------------------------------------------------------
VirtualMatrix::VirtualMatrix()
{
  // Initialize PETSc
  PETScManager::init();

  // Don't initialize the matrix
  A = 0;
}
//-----------------------------------------------------------------------------
VirtualMatrix::VirtualMatrix(const NewVector& x, const NewVector& y)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  A = 0;
  init(x, y);
}
//-----------------------------------------------------------------------------
VirtualMatrix::~VirtualMatrix()
{
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void VirtualMatrix::init(const NewVector& x, const NewVector& y)
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
      MatDestroy(A);
  }
  
  MatCreateShell(PETSC_COMM_WORLD, m, n, M, M, (void*) this, &A);
  MatShellSetOperation(A, MATOP_MULT, (void (*)()) usermult);
}
//-----------------------------------------------------------------------------
dolfin::uint VirtualMatrix::size(uint dim) const
{
  int m = 0;
  int n = 0;
  MatGetSize(A, &m, &n);
  return (dim == 0 ? static_cast<uint>(m) : static_cast<uint>(n));
}
//-----------------------------------------------------------------------------
Mat VirtualMatrix::mat()
{
  return A;
}
//-----------------------------------------------------------------------------
const Mat VirtualMatrix::mat() const
{
  return A;
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
