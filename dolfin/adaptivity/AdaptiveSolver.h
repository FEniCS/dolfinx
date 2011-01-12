// Copyright (C) 2010 Marie E. Rognes.
// Licensed under the GNU LGPL Version 3.0 or any later version.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-08-19
// Last changed: 2011-01-12

#ifndef __ADAPTIVE_SOLVER_H
#define __ADAPTIVE_SOLVER_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/adaptivity/AdaptiveDatum.h>

namespace dolfin
{

  class Function;
  class VariationalProblem;
  class ErrorControl;
  class GoalFunctional;
  class Form;

  /// An _AdaptiveSolver_ solves a _VariationalProblem_ adaptively to
  /// within a given error tolerance with respect to the error in a
  /// given _GoalFunctional_.

  class AdaptiveSolver
  {
  public:

    /// Solve given _VariationalProblem_ with respect to given
    /// _GoalFunctional_
    ///
    /// *Arguments*
    ///     u (_Function_)
    ///         the solution
    ///
    ///     problem (_VariationalProblem_)
    ///         the variational problem
    ///
    ///     M (_GoalFunctional_)
    ///         a goal functional
    ///
    ///     tol (double)
    ///         the prescribed tolerance
    static void solve(Function& u,
                      VariationalProblem& problem,
                      double tol,
                      GoalFunctional& M,
                      Parameters& parameters);

    static void solve(Function& u,
                      VariationalProblem& problem,
                      double tol,
                      Form& M,
                      ErrorControl& ec,
                      Parameters& parameters);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("adaptive_solver");

      // Set default adaptive parameters associated with stopping
      // criteria
      p.add("tolerance", 0.0);
      p.add("max_iterations", 1);
      p.add("max_dimension", 0);

      // Set generic adaptive parameters
      p.add("plot_mesh", true);
      p.add("reference", 0.0);

      // FIXME: Should nest the various parameters in a sensible way.
      // Set (dolfin) parameters for error estimation. Other
      // parameters for error estimation should be controlled by the
      // form compiler.
      p.add("dual_solver", "extrapolation");

      // Set parameters for mesh marking
      p.add("marking_strategy", "dorfler");
      p.add("marking_fraction", 0.5, 0.0, 1.0);

      return p;
    }

  private:

    // Check if stopping criterion is satisfied
    static bool stop(const FunctionSpace& V,
                     double error_estimate,
                     const Parameters& parameters);

    // Present summary of adaptive data
    static void summary(const std::vector<AdaptiveDatum>& data,
                        const Parameters& parameters);

  };

}

#endif
