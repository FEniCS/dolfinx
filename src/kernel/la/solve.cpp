// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-30
// Last changed: 2007-04-30

#include <dolfin/PETScLUSolver.h>
#include <dolfin/uBlasLUSolver.h>
#include <dolfin/solve.h>

using namespace dolfin;

#ifdef HAVE_PETSC_H
//-----------------------------------------------------------------------------
void dolfin::solve(const PETScMatrix& A,
                   PETScVector& x,
                   const PETScVector& b)
{
  PETScLUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------  
void dolfin::solve(const PETScKrylovMatrix& A,
                   PETScVector& x,
                   const PETScVector& b)
{
  PETScLUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
#endif

//-----------------------------------------------------------------------------
void dolfin::solve(const uBlasMatrix<ublas_dense_matrix>& A,
                   uBlasVector& x,
                   const uBlasVector& b)
{
  uBlasLUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const uBlasMatrix<ublas_sparse_matrix>& A,
                   uBlasVector& x,
                   const uBlasVector& b)
{
  uBlasLUSolver solver;
  solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
