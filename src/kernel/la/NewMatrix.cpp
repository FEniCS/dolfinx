// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewMatrix::NewMatrix()
{
  // Initialize PETSc
  PETScManager::init();

  // Don't initialize the matrix
  A = 0;
}
//-----------------------------------------------------------------------------
NewMatrix::NewMatrix(uint m, uint n)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  A = 0;
  init(m, n);
}
//-----------------------------------------------------------------------------
NewMatrix::~NewMatrix()
{
  // Free memory for PETSc matrix
  CHKERRQ(MatDestroy(A));
}
//-----------------------------------------------------------------------------
void NewMatrix::init(uint m, uint n)
{
  if ( A )
    CHKERRQ(MatDestroy(A));
  CHKERRQ(MatCreate(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, m, n, &A));
}
//-----------------------------------------------------------------------------
dolfin::uint NewMatrix::size(uint dim) const
{
  int m = 0;
  int n = 0;
  CHKERRQ(MatGetSize(A, &m, &n));
  return (dim == 0 ? m : n);
}
//-----------------------------------------------------------------------------
NewMatrix& NewMatrix::operator= (real zero)
{
  if ( zero != 0.0 )
    dolfin_error("Argument must be zero.");
  CHKERRQ(MatZeroEntries(A));
  return *this;
}
//-----------------------------------------------------------------------------
void NewMatrix::add(real block[], uint rows[], uint m, uint cols[], uint n)
{
  CHKERRQ(MatSetValues(A, m, rows, n, cols, block, ADD_VALUES));
}
//-----------------------------------------------------------------------------
void NewMatrix::apply()
{
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
}
//-----------------------------------------------------------------------------
Mat NewMatrix::mat()
{
  return A;
}
//-----------------------------------------------------------------------------
