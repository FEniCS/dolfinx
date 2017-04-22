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
//
// First added:  2010-08-19
// Last changed: 2012-11-14

#ifndef __GENERIC_ADAPTIVE_VARIATIONAL_SOLVER_H
#define __GENERIC_ADAPTIVE_VARIATIONAL_SOLVER_H

#include <vector>
#include <memory>
#include <dolfin/common/Variable.h>
#include <dolfin/adaptivity/ErrorControl.h>

namespace dolfin
{
  // Forward declarations
  class DirichletBC;
  class Form;
  class Function;
  class FunctionSpace;
  class GoalFunctional;
  class Mesh;
  class Parameters;

  /// An abstract class for goal-oriented adaptive solution of
  /// variational problems.
  ///
  class GenericAdaptiveVariationalSolver : public Variable
  {
  public:

    virtual ~GenericAdaptiveVariationalSolver();

    /// Solve such that the functional error is less than the given
    /// tolerance. Note that each call to solve is based on the
    /// leaf-node of the variational problem
    ///
    /// *Arguments*
    ///     tol  (double)
    ///         The error tolerance
    void solve(const double tol);

    /// Solve the primal problem. Must be overloaded in subclass.
    ///
    /// *Returns*
    ///     _Function_
    ///         The solution to the primal problem
    virtual std::shared_ptr<const Function> solve_primal() = 0;

    /// Extract the boundary conditions for the primal problem. Must
    /// be overloaded in subclass.
    ///
    /// *Returns*
    ///     std::vector<_DirichletBC_>
    ///         The primal boundary conditions
    virtual std::vector<std::shared_ptr<const DirichletBC> >
      extract_bcs() const = 0;

    /// Evaluate the goal functional. Must be overloaded in subclass.
    ///
    /// *Arguments*
    ///    M (_Form_)
    ///        The functional to be evaluated
    ///    u (_Function_)
    ///        The function of which to evaluate the functional
    ///
    /// *Returns*
    ///     double
    ///         The value of M evaluated at u
    virtual double evaluate_goal(Form& M,
                                 std::shared_ptr<const Function> u) const = 0;

    /// Adapt the problem to other mesh. Must be overloaded in subclass.
    ///
    /// *Arguments*
    ///    mesh (_Mesh_)
    ///        The other mesh
    virtual void adapt_problem(std::shared_ptr<const Mesh> mesh) = 0;

    /// Return stored adaptive data
    ///
    /// *Returns*
    ///    std::vector<_Parameters_>
    ///        The data stored in the adaptive loop
    std::vector<std::shared_ptr<Parameters> > adaptive_data() const;

    /// Default parameter values:
    ///
    ///     "max_iterations"     (int)
    ///     "max_dimension"      (int)
    ///     "plot_mesh"          (bool)
    ///     "save_data"          (bool)
    ///     "data_label"         (std::string)
    ///     "reference"          (double)
    ///     "marking_strategy"   (std::string)
    ///     "marking_fraction"   (double)
    static Parameters default_parameters()
    {
      Parameters p("adaptive_solver");

      // Set default generic adaptive parameters
      p.add("max_iterations", 50);
      p.add("max_dimension", 0);
      p.add("plot_mesh", false); // Useful for debugging
      p.add("save_data", false);
      p.add("data_label", "default/adaptivity");
      p.add("reference", 0.0);
      p.add("marking_strategy", "dorfler");
      p.add("marking_fraction", 0.5, 0.0, 1.0);

      // Set parameters for dual solver
      Parameters ec_params(ErrorControl::default_parameters());
      p.add(ec_params);

      return p;
    }

    /// Present summary of all adaptive data and parameters
    void summary();

  protected:

    /// The goal functional
    std::shared_ptr<Form> goal;

    /// Error control object
    std::shared_ptr<ErrorControl> control;

    // A list of adaptive data
    std::vector<std::shared_ptr<Parameters> > _adaptive_data;

    /// Return the number of degrees of freedom for primal problem
    ///
    /// *Returns*
    ///     _std::size_t_
    ///         The number of degrees of freedom
    virtual std::size_t num_dofs_primal() = 0;

  };
}



#endif
