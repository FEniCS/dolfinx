// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/PETScManager.h>
#include <dolfin/NewMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewMatrix::NewMatrix(unsigned int m, unsigned int n)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc matrix
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF, m, n, 10, PETSC_NULL, &A));
}
//-----------------------------------------------------------------------------
NewMatrix::~NewMatrix()
{
  // Free memory for PETSc matrix
  CHKERRQ(MatDestroy(A));
}
//-----------------------------------------------------------------------------
unsigned int NewMatrix::size(unsigned int dim) const
{
  int m = 0;
  int n = 0;
  CHKERRQ(MatGetSize(A, &m, &n));
  return (dim == 0 ? m : n);
}
//-----------------------------------------------------------------------------
Mat NewMatrix::mat()
{
  return A;
}
//-----------------------------------------------------------------------------
