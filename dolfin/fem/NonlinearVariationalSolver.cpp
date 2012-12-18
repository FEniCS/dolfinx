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
std::pair<std::size_t, bool>  NonlinearVariationalSolver::solve()
{
  begin("Solving nonlinear variational problem.");

  // Check that the Jacobian has been defined
  dolfin_assert(problem);
  if (!problem->has_jacobian())
    dolfin_error("NonlinearVariationalSolver.cpp",
                 "solve nonlinear variational problem",
                 "The Jacobian form has not been defined");

  // Get problem data
  dolfin_assert(problem);
  boost::shared_ptr<Function> u(problem->solution());

  const bool reset_jacobian = parameters["reset_jacobian"];

  // Create nonlinear problem
  if (!nonlinear_problem || reset_jacobian)
  {
    nonlinear_problem = boost::shared_ptr<NonlinearDiscreteProblem>(new NonlinearDiscreteProblem(problem,
                                             reference_to_no_delete_pointer(*this)));
  }


  std::pair<std::size_t, bool> ret;

  if (std::string(parameters["nonlinear_solver"]) == "newton")
  {
    // Create Newton solver and set parameters
    if (newton_solver || reset_jacobian)
    {
      // Create Newton solver and set parameters
      newton_solver = boost::shared_ptr<NewtonSolver>(new NewtonSolver(parameters["linear_solver"],
                                                          parameters["preconditioner"]));
    }
    newton_solver->parameters.update(parameters("newton_solver"));

    // Solve nonlinear problem using Newton's method
    dolfin_assert(u->vector());
    dolfin_assert(nonlinear_problem);
    ret = newton_solver->solve(*nonlinear_problem, *u->vector());
  }
#ifdef HAS_PETSC
  else if (std::string(parameters["nonlinear_solver"]) == "snes")
  {
    // Create SNES solver and set parameters
    if (snes_solver || reset_jacobian)
    {
      // Create Newton solver and set parameters
      snes_solver = boost::shared_ptr<PETScSNESSolver>(new PETScSNESSolver());
    }
    snes_solver->parameters.update(parameters("snes_solver"));
    snes_solver->set_linear_solver_parameters(parameters);

    // Solve nonlinear problem using PETSc's SNES
    dolfin_assert(u->vector());
    dolfin_assert(nonlinear_problem);
    ret = snes_solver->solve(*nonlinear_problem, *u->vector());
  }
#endif
  else
  {
    dolfin_error("NonlinearVariationalSolver.cpp",
                 "solve nonlinear variational problem",
                 "Unknown nonlinear solver type");
  }

  end();
  return ret;
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
  dolfin_assert(problem);
  boost::shared_ptr<const Form> F(problem->residual_form());
  std::vector<boost::shared_ptr<const BoundaryCondition> > bcs(problem->bcs());

  // Assemble right-hand side
  dolfin_assert(F);
  assemble(b, *F);

  // Apply boundary conditions
  for (std::size_t i = 0; i < bcs.size(); i++)
  {
    dolfin_assert(bcs[i]);
    bcs[i]->apply(b, x);
  }

  // Print vector
  dolfin_assert(solver);
  const bool print_rhs = solver->parameters["print_rhs"];
  if (print_rhs)
    info(b, true);
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
NonlinearDiscreteProblem::J(GenericMatrix& A, const GenericVector& x)
{
  // Get problem data
  dolfin_assert(problem);
  boost::shared_ptr<const Form> J(problem->jacobian_form());
  std::vector<boost::shared_ptr<const BoundaryCondition> > bcs(problem->bcs());

  // Check if Jacobian matrix sparsity pattern should be reset
  dolfin_assert(solver);
  const bool reset_jacobian = solver->parameters["reset_jacobian"];
  bool reset_sparsity = reset_jacobian || !jacobian_initialized;

  // Assemble left-hand side
  dolfin_assert(J);
  assemble(A, *J, reset_sparsity);

  // Remember that Jacobian has been initialized
  jacobian_initialized = true;

  // Apply boundary conditions
  for (std::size_t i = 0; i < bcs.size(); i++)
  {
    dolfin_assert(bcs[i]);
    bcs[i]->apply(A);
  }

  // Print matrix
  dolfin_assert(solver);
  const bool print_matrix = solver->parameters["print_matrix"];
  if (print_matrix)
    info(A, true);
}
//-----------------------------------------------------------------------------
