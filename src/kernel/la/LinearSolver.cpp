// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2006-06-07

#include <dolfin/LinearSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearSolver::LinearSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LinearSolver::~LinearSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, const uBlasVector& b)
{
  dolfin_error("Linear solver not available through default interface for this matrix/vector type");
  return 0;
}
//-----------------------------------------------------------------------------
#ifdef HAVE_PETSC_H
dolfin::uint LinearSolver::solve(const PETScSparseMatrix& A, PETScVector& x, const PETScVector& b)
{
  dolfin_error("Linear solver not available through default interface for this matrix/vector type");
  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(const VirtualMatrix& A, PETScVector& x, const PETScVector& b)
{
  dolfin_error("Linear solver not available through default interface for this matrix/vector type");
  return 0;
}
#endif
//-----------------------------------------------------------------------------
