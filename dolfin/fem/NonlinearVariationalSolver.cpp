// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Marie E. Rognes, 2011.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem.cpp)
// Last changed: 2011-03-29

#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/function/Function.h>
#include "assemble.h"
#include "Form.h"
#include "NonlinearVariationalProblem.h"
#include "NonlinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
NonlinearVariationalSolver(NonlinearVariationalProblem& problem)
  : problem(reference_to_no_delete_pointer(problem))
{
  // Set parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
NonlinearVariationalSolver(boost::shared_ptr<NonlinearVariationalProblem> problem)
  : problem(problem)
{
  // Set parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::solve()
{
  begin("Solving nonlinear variational problem.");

  // Check that the Jacobian has been defined
  assert(problem);
  if (!problem->has_jacobian())
    dolfin_error("NonlinearVariationalSolver.cpp",
                 "solve nonlinear variational problem",
                 "the Jacobian form has not been defined.");

  // Get problem data
  assert(problem);
  boost::shared_ptr<Function> u(problem->solution());

  // Create nonlinear problem
  NonlinearDiscreteProblem nonlinear_problem(problem,
                                             reference_to_no_delete_pointer(*this));

  // Create Newton solver and set parameters
  NewtonSolver newton_solver(parameters["linear_solver"],
                             parameters["preconditioner"]);
  newton_solver.parameters.update(parameters("newton_solver"));

  // Solve nonlinear problem using Newton's method
  assert(u);
  newton_solver.solve(nonlinear_problem, u->vector());

  end();
}
//-----------------------------------------------------------------------------
// Implementation of NonlinearDiscreteProblem
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
NonlinearDiscreteProblem::
NonlinearDiscreteProblem(boost::shared_ptr<NonlinearVariationalProblem> problem,
                         boost::shared_ptr<NonlinearVariationalSolver> solver)
  : problem(problem), solver(solver), jacobian_initialized(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
NonlinearDiscreteProblem::~NonlinearDiscreteProblem()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
NonlinearDiscreteProblem::F(GenericVector& b, const GenericVector& x)
{
  // Get problem data
  assert(problem);
  boost::shared_ptr<const Form> F(problem->residual_form());
  std::vector<boost::shared_ptr<const BoundaryCondition> > bcs(problem->bcs());

  // Assemble right-hand side
  assert(F);
  assemble(b, *F);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
  {
    assert(bcs[i]);
    bcs[i]->apply(b, x);
  }

  // Print vector
  assert(solver);
  const bool print_rhs = solver->parameters["print_rhs"];
  if (print_rhs)
    info(b, true);
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
NonlinearDiscreteProblem::J(GenericMatrix& A, const GenericVector& x)
{
  // Get problem data
  assert(problem);
  boost::shared_ptr<const Form> J(problem->jacobian_form());
  std::vector<boost::shared_ptr<const BoundaryCondition> > bcs(problem->bcs());

  // Check if Jacobian matrix sparsity pattern should be reset
  assert(solver);
  bool reset_sparsity = !(solver->parameters["reset_jacobian"] &&
                          jacobian_initialized);

  // Assemble left-hand side
  assert(J);
  assemble(A, *J, reset_sparsity);

  // Remember that Jacobian has been initialized
  jacobian_initialized = true;

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
  {
    assert(bcs[i]);
    bcs[i]->apply(A);
  }

  // Print matrix
  assert(solver);
  const bool print_matrix = solver->parameters["print_matrix"];
  if (print_matrix)
    info(A, true);
}
//-----------------------------------------------------------------------------
