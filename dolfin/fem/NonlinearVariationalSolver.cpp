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
#include "Form.h"
#include "VariationalProblem.h"
#include "NonlinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
NonlinearVariationalSolver(const Form& F,
                           const Form& J,
                           Function& u,
                           std::vector<const BoundaryCondition*> bcs)
  : F(reference_to_no_delete_pointer(F)),
    J(reference_to_no_delete_pointer(J)),
    u(reference_to_no_delete_pointer(u))
{
  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); ++i)
    this->bcs.push_back(reference_to_no_delete_pointer(*bcs[i]));

  // Set parameters
  parameters = default_parameters();

  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
NonlinearVariationalSolver(boost::shared_ptr<const Form> F,
                           boost::shared_ptr<const Form> J,
                           boost::shared_ptr<Function> u,
                           std::vector<boost::shared_ptr<const BoundaryCondition> > bcs)
  : F(F), J(J), u(u)
{
  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); ++i)
    this->bcs.push_back(bcs[i]);

  // Set parameters
  parameters = default_parameters();

  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::solve()
{
  begin("Solving nonlinear variational problem.");

  // Create nonlinear problem
  _NonlinearProblem nonlinear_problem(*this);

  // Create Newton solver and set parameters
  NewtonSolver newton_solver(parameters["linear_solver"],
                             parameters["preconditioner"]);
  newton_solver.parameters.update(parameters("newton_solver"));

  // Solve nonlinear problem using Newton's method
  newton_solver.solve(nonlinear_problem, u->vector());

  end();
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::check_forms() const
{
  // Check rank of residual F
  if (F->rank() != 1)
    dolfin_error("NonlinearVariationalSolver.cpp",
                 "create nonlinear variational solver for F(u; v) = 0 for all v",
                 "expecting the residual F to be a linear form (not rank %d).",
                 F->rank());

  // Check rank of Jacobian J
  if (J->rank() != 2)
    dolfin_error("NonlinearVariationalSolver.cpp",
                 "create nonlinear variational solver for F(u; v) = 0 for all v",
                 "expecting the Jacobian to be a bilinear form (not rank %d).",
                 J->rank());
}
//-----------------------------------------------------------------------------
// Implementation of _NonlinearProblem
//-----------------------------------------------------------------------------
NonlinearVariationalSolver::
_NonlinearProblem::_NonlinearProblem(const NonlinearVariationalSolver& solver)
  : solver(solver), jacobian_initialized(false)
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
  // Assemble right-hand side
  assert(solver.F);
  assemble(b, *solver.F);

  // Apply boundary conditions
  for (uint i = 0; i < solver.bcs.size(); i++)
    solver.bcs[i]->apply(b, x);

  // Print vector
  const bool print_rhs = solver.parameters["print_rhs"];
  if (print_rhs == true)
    info(b, true);
}
//-----------------------------------------------------------------------------
void NonlinearVariationalSolver::
_NonlinearProblem::J(GenericMatrix& A, const GenericVector& x)
{
  // Check if Jacobian matrix sparsity pattern should be reset
  bool reset_sparsity = !(solver.parameters["reset_jacobian"] &&
                          jacobian_initialized);

  // Assemble left-hand side
  assert(solver.J);
  assemble(A, *solver.J, reset_sparsity);

  // Remember that Jacobian has been initialized
  jacobian_initialized = true;

  // Apply boundary conditions
  for (uint i = 0; i < solver.bcs.size(); i++)
    solver.bcs[i]->apply(A);

  // Print matrix
  const bool print_matrix = solver.parameters["print_matrix"];
  if (print_matrix == true)
    info(A, true);
}
//-----------------------------------------------------------------------------
