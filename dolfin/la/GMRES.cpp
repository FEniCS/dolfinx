// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
// 
// First added:  2006-08-16
// Last changed: 2006-08-16

#include "GMRES.h"

using namespace dolfin;

#ifdef HAS_PETSC
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const PETScMatrix& A, PETScVector& x,
			  const PETScVector& b, Preconditioner pc)
{
  PETScKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const PETScKrylovMatrix& A, PETScVector& x,
			  const PETScVector& b, Preconditioner pc)
{
  PETScKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const PETScMatrix& A, PETScVector& x,
			  const PETScVector& b, PETScPreconditioner& pc)
{
  PETScKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const PETScKrylovMatrix& A, PETScVector& x,
			  const PETScVector& b, PETScPreconditioner& pc)
{
  PETScKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
#endif

//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const uBlasMatrix<ublas_dense_matrix>& A,
			  uBlasVector& x, const uBlasVector& b,
			  Preconditioner pc)
{
  uBlasKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const uBlasMatrix<ublas_sparse_matrix>& A,
			  uBlasVector& x, const uBlasVector& b,
			  Preconditioner pc)
{
  uBlasKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const uBlasKrylovMatrix& A, uBlasVector& x,
			  const uBlasVector& b, Preconditioner pc)
{
  uBlasKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const uBlasMatrix<ublas_dense_matrix>& A,
			  uBlasVector& x, const uBlasVector& b,
			  uBlasPreconditioner& pc)
{
  uBlasKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const uBlasMatrix<ublas_sparse_matrix>& A,
			  uBlasVector& x, const uBlasVector& b,
			  uBlasPreconditioner& pc)
{
  uBlasKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint GMRES::solve(const uBlasKrylovMatrix& A, uBlasVector& x,
			  const uBlasVector& b, uBlasPreconditioner& pc)
{
  uBlasKrylovSolver solver(gmres, pc);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
