// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-16
// Last changed: 2006-08-16

#include "LU.h"

using namespace dolfin;

#ifdef HAS_PETSC
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
