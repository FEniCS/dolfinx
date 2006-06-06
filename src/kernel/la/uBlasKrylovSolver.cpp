// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed: 2006-06-06

#include <dolfin/dolfin_log.h>
#include <dolfin/uBlasKrylovSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::uBlasKrylovSolver(Type solver) : type(solver)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasKrylovSolver::~uBlasKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint uBlasKrylovSolver::solve(const uBlasSparseMatrix& A, DenseVector& x, 
    const DenseVector& b)
{
  dolfin_warning("Krylov solvers for uBlas data types have not been implemented.");
  dolfin_warning("LU solver will be used. This may be slow and consume a lot of memory.");
        
  // FIXME: implement renumbering scheme to speed up LU solve
  A.solve(x, b);
  return 1;
}
//-----------------------------------------------------------------------------
