// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewVector.h>
#include <dolfin/VirtualMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
VirtualMatrix::VirtualMatrix()
{
  // Initialize PETSc
  PETScManager::init();

  // Don't initialize the matrix
  A = 0;
}
//-----------------------------------------------------------------------------
VirtualMatrix::VirtualMatrix(const NewVector& y)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  A = 0;
  init(y);
}
//-----------------------------------------------------------------------------
VirtualMatrix::~VirtualMatrix()
{
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void VirtualMatrix::init(const NewVector& y)
{
/*
  // Get size and local size of given vector and existing matrix
  int m = 0;
  int M = 0;
  VecGetLocalSize(y, &m);
  VecGetSize(y, &M);

  // Free previously allocated memory if necessary
  if ( A )
  {
    // Get size and local size of existing matrix
    int mm(0), M(0);


      if ( M == size(0) && m == size(1) )
      return;
    else
      MatDestroy(A);

    }

  
  MatCreateShell(comm,m,N,M,N,ctx,&A);
  MatShellSetOperation(mat,MAT_MULT,mult); 
 
  MatCreate(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, &A);
  MatSetFromOptions(A);
  */
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
