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
// Modified by Anders Logg, 2010-2011.
// Modified by Garth N. Wells, 2011.
//
// First added:  2010-08-19
// Last changed: 2012-11-14

#ifndef __ADAPTIVE_LINEAR_VARIATIONAL_SOLVER_H
#define __ADAPTIVE_LINEAR_VARIATIONAL_SOLVER_H

#include <memory>
#include "GenericAdaptiveVariationalSolver.h"

namespace dolfin
{
  // Forward declarations
  class DirichletBC;
  class Form;
  class Function;
  class LinearVariationalProblem;
  class GoalFunctional;
  class Mesh;

  /// A class for goal-oriented adaptive solution of linear
  /// variational problems.
  ///
  /// For a linear variational problem of the form: find u in V
  /// satisfying
  ///
  ///     a(u, v) = L(v) for all v in :math:`\hat V`
  ///
  /// and a corresponding conforming discrete problem: find u_h in V_h
  /// satisfying
  ///
  ///     a(u_h, v) = L(v) for all v in :math:`\hat V_h`
  ///
  /// and a given goal functional M and tolerance tol, the aim is to
  /// find a V_H and a u_H in V_H satisfying the discrete problem such
  /// that
  ///
  ///     \|M(u) - M(u_H)\| < tol
  ///
  /// This strategy is based on dual-weighted residual error
  /// estimators designed and automatically generated for the primal
  /// problem and subsequent h-adaptivity.

  class AdaptiveLinearVariationalSolver
    : public GenericAdaptiveVariationalSolver
  {
  public:

    /// Create AdaptiveLinearVariationalSolver (shared ptr version)
    ///
    /// *Arguments*
    ///     problem (_LinearVariationalProblem_)
    ///         The primal problem
    ///     goal (_GoalFunctional_)
    ///         The goal functional
    AdaptiveLinearVariationalSolver(std::shared_ptr<LinearVariationalProblem> problem,
                                    std::shared_ptr<GoalFunctional> goal);

    /// Create AdaptiveLinearVariationalSolver from variational
    /// problem, goal form and error control instance
    ///
    /// *Arguments*
    ///     problem (_LinearVariationalProblem_)
    ///         The primal problem
    ///     goal (_Form_)
    ///         The goal functional
    ///     control (_ErrorControl_)
    ///         An error controller object
    AdaptiveLinearVariationalSolver(std::shared_ptr<LinearVariationalProblem> problem,
                                    std::shared_ptr<Form> goal,
                                    std::shared_ptr<ErrorControl> control);

    /// Destructor
    ~AdaptiveLinearVariationalSolver() {}

    /// Solve the primal problem.
    ///
    /// *Returns*
    ///     _Function_
    ///         The solution to the primal problem
    virtual std::shared_ptr<const Function> solve_primal();

    /// Extract the boundary conditions for the primal problem.
    ///
    /// *Returns*
    ///     std::vector<_DirichletBC_>
    ///         The primal boundary conditions
    virtual std::vector<std::shared_ptr<const DirichletBC>> extract_bcs() const;

    /// Evaluate the goal functional.
    ///
    /// *Arguments*
    ///    M (_Form_)
    ///        The functional to be evaluated
    ///    u (_Function_)
    ///        The function at which to evaluate the functional
    ///
    /// *Returns*
    ///     double
    ///         The value of M evaluated at u
    virtual double evaluate_goal(Form& M,
                                 std::shared_ptr<const Function> u) const;

    /// Adapt the problem to other mesh.
    ///
    /// *Arguments*
    ///    mesh (_Mesh_)
    ///        The other mesh
    virtual void adapt_problem(std::shared_ptr<const Mesh> mesh);

  protected:

    /// Return the number of degrees of freedom for primal problem
    ///
    /// *Returns*
    ///     _std::size_t_
    ///         The number of degrees of freedom
    virtual std::size_t num_dofs_primal();

  private:

    /// Helper function for instance initialization
    ///
    /// *Arguments*
    ///    problem (_LinearVariationalProblem_)
    ///        The primal problem
    ///    u (_GoalFunctional_)
    ///        The goal functional
    void init(std::shared_ptr<LinearVariationalProblem> problem,
              std::shared_ptr<GoalFunctional> goal);

    // The primal problem
    std::shared_ptr<LinearVariationalProblem> _problem;

  };

}

#endif
