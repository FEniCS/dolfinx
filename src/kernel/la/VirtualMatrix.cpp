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
VirtualMatrix::VirtualMatrix(uint m, uint n)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  A = 0;
  init(m, n);
}
//-----------------------------------------------------------------------------
VirtualMatrix::~VirtualMatrix()
{
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void VirtualMatrix::init(uint m, uint n)
{
  /*
  // Free previously allocated memory if necessary
  if ( A )
    if ( m == size(0) && n == size(1) )
      return;
    else
      MatDestroy(A);
  
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
