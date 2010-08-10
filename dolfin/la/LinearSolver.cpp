// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hake, 2010.
// Modified by Garth N. Wells, 2010.
//
// First added:  2008-05-10
// Last changed: 2010-07-16

#include "KrylovSolver.h"
#include "LUSolver.h"
#include "LinearSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearSolver::LinearSolver(std::string solver_type, std::string pc_type)
{
  // Choose solver and set parameters
  if (solver_type == "lu" || solver_type == "cholesky")
    solver.reset(new LUSolver(solver_type));
  else
    solver.reset(new KrylovSolver(solver_type, pc_type));

  // Get parameters
  parameters = solver->parameters;

  // Rename the parameters
  parameters.rename("linear_solver");
}
//-----------------------------------------------------------------------------
LinearSolver::~LinearSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LinearSolver::set_operator(const GenericMatrix& A)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
void LinearSolver::set_operators(const GenericMatrix& A, const GenericMatrix& P)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operators(A, P);
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(const GenericMatrix& A, GenericVector& x,
                                 const GenericVector& b)
{
  assert(solver);
  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(solver);
  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------

