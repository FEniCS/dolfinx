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
NewMatrix::NewMatrix(int m, int n)
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
  // Free memory of matrix
  if ( A ) MatDestroy(A);
}
//-----------------------------------------------------------------------------
void NewMatrix::init(int m, int n)
{
  // Free previously allocated memory if necessary
  if ( A )
    if ( m == size(0) && n == size(1) )
      return;
    else
      MatDestroy(A);
  
  MatCreate(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, n, &A);
  MatSetFromOptions(A);
}
//-----------------------------------------------------------------------------
int NewMatrix::size(int dim) const
{
  int m = 0;
  int n = 0;
  MatGetSize(A, &m, &n);
  return (dim == 0 ? m : n);
}
//-----------------------------------------------------------------------------
NewMatrix& NewMatrix::operator= (real zero)
{
  if ( zero != 0.0 )
    dolfin_error("Argument must be zero.");
  MatZeroEntries(A);
  return *this;
}
//-----------------------------------------------------------------------------
void NewMatrix::add(const real block[],
		    const int rows[], int m,
		    const int cols[], int n)
{
  MatSetValues(A, m, rows, n, cols, block, ADD_VALUES);
}
//-----------------------------------------------------------------------------
void NewMatrix::apply()
{
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}
//-----------------------------------------------------------------------------
Mat NewMatrix::mat()
{
  return A;
}
//-----------------------------------------------------------------------------
const Mat NewMatrix::mat() const
{
  return A;
}
//-----------------------------------------------------------------------------
void NewMatrix::disp() const
{
  MatView(A, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
LogStream& dolfin::operator<< (LogStream& stream, const NewMatrix& A)
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
