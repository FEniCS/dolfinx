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
// Last changed: 2011-03-31

#ifndef __ADAPTIVE_LINEAR_VARIATIONAL_SOLVER_H
#define __ADAPTIVE_LINEAR_VARIATIONAL_SOLVER_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/adaptivity/AdaptiveDatum.h>

namespace dolfin
{
  // Forward declarations
  class Parameters;
  class FunctionSpace;
  class LinearVariationalProblem;
  class NonlinearVariationalProblem;
  class GoalFunctional;
  class ErrorControl;

  /// An _AdaptiveLinearVariationalSolver_ solves a
  /// _LinearVariationalProblem_ or a _NonlinearVariationalProblem_
  /// adaptively to within a given error tolerance with respect to the
  /// error in a given _GoalFunctional_.

  class AdaptiveLinearVariationalSolver : public Variable
  {
  public:

    /// Create adaptive variational solver for given linear variaional
    /// problem
    AdaptiveLinearVariationalSolver(LinearVariationalProblem& problem);

    /// Create adaptive variational solver for given nonlinear
    /// variaional problem
    AdaptiveLinearVariationalSolver(NonlinearVariationalProblem& problem);

    /// Solve given _LinearVariationalProblem_ with respect to given
    /// _GoalFunctional_ to within the given tolerance
    ///
    /// *Arguments*
    ///
    ///     tol (double)
    ///         a prescribed error tolerance
    ///
    ///     M (_GoalFunctional_)
    ///         a goal functional
    ///
    void solve(const double tol, GoalFunctional& M);

    /// Solve given _LinearVariationalProblem_ with respect to given
    /// _GoalFunctional_ to within the given tolerance
    ///
    /// *Arguments*
    ///
    ///     tol (double)
    ///         a prescribed error tolerance
    ///
    ///     M (_GoalFunctional_)
    ///         a goal functional
    ///
    ///     ec (_ErrorControl_)
    ///         an ErrorController object
    ///
    void solve(const double tol, GoalFunctional& M, ErrorControl& ec);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("adaptive_linear_variational_solver");

      // Set default adaptive parameters
      p.add("max_iterations", 20);
      p.add("max_dimension", 0);

      // Set generic adaptive parameters
      p.add("plot_mesh", false); // Very useful for debugging
      p.add("reference", 0.0);

      // Set parameters for mesh marking
      p.add("marking_strategy", "dorfler");
      p.add("marking_fraction", 0.5, 0.0, 1.0);

      return p;
    }

  private:

    // The problem
    boost::shared_ptr<LinearVariationalProblem> problem;

    // Check if stopping criterion is satisfied
    bool stop(const FunctionSpace& V,
              const double error_estimate,
              const double tolerance,
              const Parameters& parameters);

    // Present summary of adaptive data
    void summary(const std::vector<AdaptiveDatum>& data,
                 const Parameters& parameters);

    // Present summary of adaptive data
    void summary(const AdaptiveDatum& data);

  };

}

#endif
