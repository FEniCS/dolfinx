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
#include <dolfin/fem/LinearVariationalProblem.h>
#include <dolfin/fem/LinearVariationalSolver.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>

#include "AdaptiveLinearVariationalSolver.h"
#include "GoalFunctional.h"
#include "adapt.h"

using namespace dolfin;

// ----------------------------------------------------------------------------
AdaptiveLinearVariationalSolver::
AdaptiveLinearVariationalSolver(LinearVariationalProblem& problem,
                                GoalFunctional& goal)
  : problem(reference_to_no_delete_pointer(problem))
{
  init(reference_to_no_delete_pointer(problem),
       reference_to_no_delete_pointer(goal));
}
// ----------------------------------------------------------------------------
AdaptiveLinearVariationalSolver::
AdaptiveLinearVariationalSolver(boost::shared_ptr<LinearVariationalProblem> problem,
                                boost::shared_ptr<GoalFunctional> goal)
  : problem(problem)
{
  init(problem, goal);
}
// ----------------------------------------------------------------------------
AdaptiveLinearVariationalSolver::
AdaptiveLinearVariationalSolver(boost::shared_ptr<LinearVariationalProblem> problem,
                                boost::shared_ptr<Form> goal,
                                boost::shared_ptr<ErrorControl> control)
  : problem(problem)
{
  this->goal = goal;
  this->control = control;

  // Set generic adaptive parameters
  parameters = GenericAdaptiveVariationalSolver::default_parameters();

  // Add parameters for linear variational solver
  parameters.add(LinearVariationalSolver::default_parameters());
}
// ----------------------------------------------------------------------------
void AdaptiveLinearVariationalSolver::
init(boost::shared_ptr<LinearVariationalProblem> problem,
     boost::shared_ptr<GoalFunctional> goal)
{
  this->goal = goal;

  // Set generic adaptive parameters
  parameters = GenericAdaptiveVariationalSolver::default_parameters();

  // Add parameters for linear variational solver
  parameters.add(LinearVariationalSolver::default_parameters());

  // Extract error control from goal
  boost::shared_ptr<const Form> a = problem->bilinear_form();
  boost::shared_ptr<const Form> L = problem->linear_form();
  dolfin_assert(a);
  dolfin_assert(L);

  // Extract and set error control from goal functional
  goal->update_ec(*a, *L);
  control = goal->_ec;
}
// ----------------------------------------------------------------------------
boost::shared_ptr<const Function>
AdaptiveLinearVariationalSolver::solve_primal()
{
  LinearVariationalProblem& current = problem->leaf_node();
  LinearVariationalSolver solver(current);
  solver.parameters.update(parameters("linear_variational_solver"));
  solver.solve();
  return current.solution();
}
// ----------------------------------------------------------------------------
std::vector<boost::shared_ptr<const BoundaryCondition> >
AdaptiveLinearVariationalSolver::extract_bcs() const
{
  const LinearVariationalProblem& current = problem->leaf_node();
  return current.bcs();
}
// ----------------------------------------------------------------------------
double AdaptiveLinearVariationalSolver::
evaluate_goal(Form& M, boost::shared_ptr<const Function> u) const
{
  dolfin_assert(M.num_coefficients() > 0);
  M.set_coefficient(M.num_coefficients() - 1, u);
  return assemble(M);
}
// ----------------------------------------------------------------------------
void AdaptiveLinearVariationalSolver::
adapt_problem(boost::shared_ptr<const Mesh> mesh)
{
  const LinearVariationalProblem& current = problem->leaf_node();
  adapt(current, mesh);
}
// ----------------------------------------------------------------------------
std::size_t AdaptiveLinearVariationalSolver::num_dofs_primal()
{
  const LinearVariationalProblem& current = problem->leaf_node();
  const FunctionSpace& V = *(current.trial_space());
  return V.dim();
}
// ----------------------------------------------------------------------------
