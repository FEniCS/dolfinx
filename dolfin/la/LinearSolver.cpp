// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-10
// Last changed: 2009-06-30

#include "LinearSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearSolver::LinearSolver(std::string solver_type, std::string pc_type)
                         : lu_solver(0), krylov_solver(0)
{
  // Set default parameters
  parameters = default_parameters();

  // Choose solver
  if (solver_type == "lu" || solver_type == "cholesky")
    lu_solver = new LUSolver(solver_type);
  else
    krylov_solver = new KrylovSolver(solver_type, pc_type);
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
  assert(lu_solver || krylov_solver);

  if (lu_solver)
    return lu_solver->solve(A, x, b);
  else
    return krylov_solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
