// Copyright (C) 2010--2012 Marie E. Rognes
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
// Modified by Anders Logg 2010-2011
// Modified by Garth N. Wells 2011
//
// First added:  2010-08-19
// Last changed: 2012-11-14

#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/NonlinearVariationalProblem.h>
#include <dolfin/fem/NonlinearVariationalSolver.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>

#include "AdaptiveNonlinearVariationalSolver.h"
#include "GoalFunctional.h"
#include "adapt.h"

using namespace dolfin;

// ----------------------------------------------------------------------------
AdaptiveNonlinearVariationalSolver::AdaptiveNonlinearVariationalSolver(
  std::shared_ptr<NonlinearVariationalProblem> problem,
  std::shared_ptr<GoalFunctional> goal)
  : _problem(problem)
{
  init(problem, goal);
}
// ----------------------------------------------------------------------------
AdaptiveNonlinearVariationalSolver::AdaptiveNonlinearVariationalSolver(
  std::shared_ptr<NonlinearVariationalProblem> problem,
  std::shared_ptr<Form> goal,
  std::shared_ptr<ErrorControl> control)
  : _problem(problem)
{
  this->goal = goal;
  this->control = control;

  // Set generic adaptive parameters
  parameters = GenericAdaptiveVariationalSolver::default_parameters();

  // Add parameters for non-linear variational solver
  parameters.add(NonlinearVariationalSolver::default_parameters());
}
// ----------------------------------------------------------------------------
void AdaptiveNonlinearVariationalSolver::init(
  std::shared_ptr<NonlinearVariationalProblem> problem,
  std::shared_ptr<GoalFunctional> goal)
{
  this->goal = goal;

  // Set generic adaptive parameters
  parameters = GenericAdaptiveVariationalSolver::default_parameters();

  // Add parameters for nonlinear variational solver
  parameters.add(NonlinearVariationalSolver::default_parameters());

  // Extract error control from goal
  std::shared_ptr<const Form> a = problem->jacobian_form();
  std::shared_ptr<const Form> L = problem->residual_form();
  dolfin_assert(a);
  dolfin_assert(L);

  // Extract error control from goal functional
  goal->update_ec(*a, *L);
  control = goal->_ec;
}
// ----------------------------------------------------------------------------
std::shared_ptr<const Function>
AdaptiveNonlinearVariationalSolver::solve_primal()
{
  NonlinearVariationalProblem& current = _problem->leaf_node();
  NonlinearVariationalSolver solver(current);
  solver.parameters.update(parameters("nonlinear_variational_solver"));
  solver.solve();
  return current.solution();
}
// ----------------------------------------------------------------------------
std::vector<std::shared_ptr<const DirichletBC>>
AdaptiveNonlinearVariationalSolver::extract_bcs() const
{
  const NonlinearVariationalProblem& current = _problem->leaf_node();
  return current.bcs();
}
// ----------------------------------------------------------------------------
double AdaptiveNonlinearVariationalSolver::evaluate_goal(Form& M,
                                 std::shared_ptr<const Function> u) const
{
  return assemble(M);
}
// ----------------------------------------------------------------------------
void AdaptiveNonlinearVariationalSolver::adapt_problem(
  std::shared_ptr<const Mesh> mesh)
{
  const NonlinearVariationalProblem& current = _problem->leaf_node();
  adapt(current, mesh);
}
// ----------------------------------------------------------------------------
std::size_t AdaptiveNonlinearVariationalSolver::num_dofs_primal()
{
  const NonlinearVariationalProblem& current = _problem->leaf_node();
  const FunctionSpace& V = *(current.trial_space());
  return V.dim();
}
// ----------------------------------------------------------------------------
