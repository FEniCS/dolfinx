// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-10
// Last changed: 2008-05-10

#include "LUSolver.h"
#include "KrylovSolver.h"
#include "LinearSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearSolver::LinearSolver(SolverType solver_type, PreconditionerType pc_type)
                         : lu_solver(0), krylov_solver(0)
{
  if (solver_type == lu)
  {
    lu_solver = new LUSolver();
    lu_solver->set("parent", *this);
  }
  else
  {
    krylov_solver = new KrylovSolver(solver_type, pc_type);
    krylov_solver->set("parent", *this);
  }
}
//-----------------------------------------------------------------------------
LinearSolver::~LinearSolver()
{
  delete lu_solver;
  delete krylov_solver;
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(const GenericMatrix& A, GenericVector& x,
                                 const GenericVector& b)
{
  dolfin_assert(lu_solver || krylov_solver);
  
  if (lu_solver)
    return lu_solver->solve(A, x, b);
  else
    return krylov_solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
