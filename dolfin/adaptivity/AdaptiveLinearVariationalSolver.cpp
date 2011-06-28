// Copyright (C) 2010 Marie E. Rognes
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
// Modified by Anders Logg, 2010-2011.
//
// First added:  2010-08-19
// Last changed: 2011-06-22

#include <dolfin/common/NoDeleter.h>
#include <dolfin/fem/LinearVariationalProblem.h>
#include <dolfin/fem/LinearVariationalSolver.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/Function.h>

#include "AdaptiveLinearVariationalSolver.h"
#include "GoalFunctional.h"
#include "adapt.h"

using namespace dolfin;

// ----------------------------------------------------------------------------
AdaptiveLinearVariationalSolver::
AdaptiveLinearVariationalSolver(LinearVariationalProblem& problem)
  : problem(reference_to_no_delete_pointer(problem))
{
  // Set generic adaptive parameters
  parameters = GenericAdaptiveVariationalSolver::default_parameters();

  // Set other parameters
  // FIXME
}
// ----------------------------------------------------------------------------
void AdaptiveLinearVariationalSolver::solve(const double tol, GoalFunctional& M)
{
  // Initialize goal functional
  boost::shared_ptr<const Form> a = problem->bilinear_form();
  boost::shared_ptr<const Form> L = problem->linear_form();
  assert(a);
  assert(L);
  M.update_ec(*a, *L);

  // Extract error control from goal functional
  assert(M._ec);
  ErrorControl& ec(*M._ec);

  // Call solve with given error control
  GenericAdaptiveVariationalSolver::solve(tol, M, ec);
}
// ----------------------------------------------------------------------------
boost::shared_ptr<const Function>
AdaptiveLinearVariationalSolver::solve_primal()
{
  LinearVariationalProblem& current = problem->fine();
  LinearVariationalSolver solver(current);
  solver.solve();
  return current.solution();
}
// ----------------------------------------------------------------------------
std::vector<boost::shared_ptr<const BoundaryCondition> >
AdaptiveLinearVariationalSolver::extract_bcs() const
{
  const LinearVariationalProblem& current = problem->fine();
  return current.bcs();
}
// ----------------------------------------------------------------------------
double AdaptiveLinearVariationalSolver::
evaluate_goal(Form& M, boost::shared_ptr<const Function> u) const
{
  assert(M.num_coefficients() > 0);
  M.set_coefficient(M.num_coefficients() - 1, u);
  return assemble(M);
}
// ----------------------------------------------------------------------------
void AdaptiveLinearVariationalSolver::
adapt_problem(boost::shared_ptr<const Mesh> mesh)
{
  const LinearVariationalProblem& current = problem->fine();
  adapt(current, mesh);
}
// ----------------------------------------------------------------------------
