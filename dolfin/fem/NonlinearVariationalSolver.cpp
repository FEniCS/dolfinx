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
// Modified by Marie E. Rognes 2011
// Modified by Corrado Maurini 2013
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem.cpp)
// Last changed: 2013-11-21

#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include "Assembler.h"
#include "Form.h"
#include "NonlinearVariationalProblem.h"
#include "SystemAssembler.h"
#include "NonlinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearVariationalSolver::NonlinearVariationalSolver(std::shared_ptr<NonlinearVariationalProblem> problem)
  : _problem(problem)
{
  // Set parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, bool> NonlinearVariationalSolver::solve()
{
  begin("Solving nonlinear variational problem.");

  // Check that the Jacobian has been defined
  dolfin_assert(_problem);
  if (!_problem->has_jacobian())
  {
    dolfin_error("NonlinearVariationalSolver.cpp",
                 "solve nonlinear variational problem",
                 "The Jacobian form has not been defined");
  }
  // Check that the dolfin is configured with petsc is bounds are set
#ifndef HAS_PETSC
  if (_problem->has_lower_bound() || _problem->has_upper_bound())
  {
    dolfin_error("NonlinearVariationalSolver.cpp",
                 "solve nonlinear variational problem",
                 "Needs PETSc to solve bound constrained problems");
  }
#endif

  // Get problem data
  dolfin_assert(_problem);
  auto u = _problem->solution();

  // Create discrete nonlinear problem
  if (!nonlinear_problem)
  {
    nonlinear_problem = std::make_shared<NonlinearDiscreteProblem>(_problem,
                                                    reference_to_no_delete_pointer(*this));
  }

  std::pair<std::size_t, bool> ret;
  if (std::string(parameters["nonlinear_solver"]) == "newton")
  {
    if (_problem->has_lower_bound() && _problem->has_upper_bound())
    {
      dolfin_error("NonlinearVariationalSolver.cpp",
                   "solve nonlinear variational problem",
                   "Set the \"nonlinear_solver\" parameter to \"snes\" or remove bounds");
    }

    // Create Newton solver and set parameters
    if (!newton_solver)
    {
      dolfin_assert(u->function_space()->mesh());
      MPI_Comm comm = u->function_space()->mesh()->mpi_comm();
      newton_solver = std::make_shared<NewtonSolver>(comm);
    }

    // Pass parameters to Newton solver
    dolfin_assert(newton_solver);
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
    if (!snes_solver)
    {
      dolfin_assert(u->function_space()->mesh());
      MPI_Comm comm = u->function_space()->mesh()->mpi_comm();
      snes_solver = std::make_shared<PETScSNESSolver>(comm);
    }
    snes_solver->parameters.update(parameters("snes_solver"));

    // Solve nonlinear problem using PETSc's SNES
    dolfin_assert(u->vector());
    dolfin_assert(nonlinear_problem);
    if (_problem->has_lower_bound() && _problem->has_upper_bound())
    {
      ret = snes_solver->solve(*nonlinear_problem, *u->vector(),
                               *_problem->lower_bound(),
                               *_problem->upper_bound());
    }
    else
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
NonlinearVariationalSolver::NonlinearDiscreteProblem::
NonlinearDiscreteProblem(std::shared_ptr<const NonlinearVariationalProblem> problem,
                         std::shared_ptr<const NonlinearVariationalSolver> solver)
  : _problem(problem), _solver(solver)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::NonlinearDiscreteProblem::~NonlinearDiscreteProblem()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
NonlinearDiscreteProblem::F(GenericVector& b, const GenericVector& x)
{
  // Get problem data
  dolfin_assert(_problem);
  std::shared_ptr<const Form> F = _problem->residual_form();
  std::shared_ptr<const Form> J = _problem->jacobian_form();
  std::vector<std::shared_ptr<const DirichletBC>> bcs(_problem->bcs());

  // Create assembler
  dolfin_assert(J);
  dolfin_assert(F);
  SystemAssembler assembler(J, F, bcs);

  // Assemble right-hand side
  assembler.assemble(b, x);

  // Print vector
  dolfin_assert(_solver);
  const bool print_rhs = _solver->parameters["print_rhs"];
  if (print_rhs)
    info(b, true);
}
//-----------------------------------------------------------------------------
void
NonlinearVariationalSolver::NonlinearDiscreteProblem::J(GenericMatrix& A,
                                                        const GenericVector& x)
{
  // Get problem data
  dolfin_assert(_problem);
  std::shared_ptr<const Form> J = _problem->jacobian_form();
  std::shared_ptr<const Form> F = _problem->residual_form();
  std::vector<std::shared_ptr<const DirichletBC>> bcs(_problem->bcs());

  // Create assembler
  dolfin_assert(J);
  dolfin_assert(F);
  SystemAssembler assembler(J, F, bcs);

  // Assemble left-hand side
  assembler.assemble(A);

  // Print matrix
  dolfin_assert(_solver);
  const bool print_matrix = _solver->parameters["print_matrix"];
  if (print_matrix)
    info(A, true);
}
//-----------------------------------------------------------------------------
