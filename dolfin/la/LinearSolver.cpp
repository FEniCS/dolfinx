// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hake, 2010.
//
// First added:  2008-05-10
// Last changed: 2010-03-04


#include "LinearSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearSolver::LinearSolver(std::string solver_type, std::string pc_type)
                         : lu_solver(0), krylov_solver(0)
{
  // Choose solver and set parameters
  if (solver_type == "lu" || solver_type == "cholesky")
  {
    lu_solver = new LUSolver(solver_type);
    parameters = lu_solver->parameters;
  }
  else
  {
    krylov_solver = new KrylovSolver(solver_type, pc_type);
    parameters = krylov_solver->parameters;
  }
  
  // Rename the parameters
  parameters.rename("linear_solver");
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
  {
    lu_solver->parameters.update(parameters);
    return lu_solver->solve(A, x, b);
  }
  else
  {
    krylov_solver->parameters.update(parameters);
    return krylov_solver->solve(A, x, b);
  }
}
//-----------------------------------------------------------------------------
