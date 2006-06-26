// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-01
// Last changed: 2006-06-06

#include <dolfin/dolfin_log.h>
#include <dolfin/uBlasLUSolver.h>
#include <dolfin/uBlasKrylovSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasLUSolver::uBlasLUSolver() : LinearSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasLUSolver::~uBlasLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasLUSolver::solve(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b)
{
//  dolfin_warning("LU solver will be used. This may be slow and consume a lot of memory.");
  dolfin_warning("Suitable LU solver for large uBlas matrices is not yet available. Krylov solver will be used.");
        
  // FIXME: find solution for decent uBlas direct solve.
//  A.solve(x, b);

  // Use Krylov solver
  uBlasKrylovSolver solver;
  solver.solve(A, x, b);

  return 1;
}
//-----------------------------------------------------------------------------
