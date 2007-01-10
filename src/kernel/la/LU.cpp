// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-08-16
// Last changed: 2006-08-16

#include <dolfin/LU.h>

using namespace dolfin;

#ifdef HAVE_PETSC_H
//-----------------------------------------------------------------------------
void LU::solve(const PETScMatrix& A, PETScVector& x,
	       const PETScVector& b)
{
  PETScLUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
void LU::solve(const PETScKrylovMatrix& A, PETScVector& x,
	       const PETScVector& b)
{
  PETScLUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
#endif

//-----------------------------------------------------------------------------
void LU::solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x,
	       const uBlasVector& b)
{
  uBlasLUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
void LU::solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x,
	       const uBlasVector& b)
{
  uBlasLUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
