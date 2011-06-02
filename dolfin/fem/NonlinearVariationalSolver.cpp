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

#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/function/Function.h>
#include "assemble.h"
#include "VariationalProblem.h"
#include "NonlinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::solve(Function& u,
                                       const VariationalProblem& problem,
                                       const Parameters& parameters)
{
  begin("Solving nonlinear variational problem.");

  // Create nonlinear problem
  _NonlinearProblem nonlinear_problem(problem, parameters);

  // Create Newton solver and set parameters
  NewtonSolver newton_solver(parameters["linear_solver"],
                             parameters["preconditioner"]);
  newton_solver.parameters.update(parameters("newton_solver"));

  // Solve nonlinear problem using Newton's method
  newton_solver.solve(nonlinear_problem, u.vector());

  end();
}
//-----------------------------------------------------------------------------
// Implementation of _NonlinearProblem
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
_NonlinearProblem::_NonlinearProblem(const VariationalProblem& problem,
                                     const Parameters& parameters)
  : problem(problem), parameters(parameters), jacobian_initialized(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
_NonlinearProblem::~_NonlinearProblem()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
_NonlinearProblem::F(GenericVector& b, const GenericVector& x)
{
  // Get F (linear form)
  boost::shared_ptr<const Form> _F = problem.linear_form();
  assert(_F);

  // Assemble right-hand side
  assemble(b, *_F);

  // Apply boundary conditions
  for (uint i = 0; i < problem.bcs().size(); i++)
    problem.bcs()[i]->apply(b, x);

  // Print vector
  const bool print_rhs = parameters["print_rhs"];
  if (print_rhs == true)
    info(b, true);
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
_NonlinearProblem::J(GenericMatrix& A, const GenericVector& x)
{
  // Check if Jacobian matrix sparsity pattern should be reset
  bool reset_sparsity = !(parameters["reset_jacobian"] && jacobian_initialized);

  // Get J (Jacobian, bilinear form)
  boost::shared_ptr<const Form> _J = problem.bilinear_form();
  assert(_J);

  // Assemble left-hand side
  assemble(A, *_J, reset_sparsity);

  // Remember that Jacobian has been initialized
  jacobian_initialized = true;

  // Apply boundary conditions
  for (uint i = 0; i < problem.bcs().size(); i++)
    problem.bcs()[i]->apply(A);

  // Print matrix
  const bool print_matrix = parameters["print_matrix"];
  if (print_matrix == true)
    info(A, true);
}
//-----------------------------------------------------------------------------
