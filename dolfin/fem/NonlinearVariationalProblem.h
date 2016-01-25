// Copyright (C) 2011 Anders Logg
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
// Modified by Corrado Maurini, 2013.

#ifndef __NONLINEAR_VARIATIONAL_PROBLEM_H
#define __NONLINEAR_VARIATIONAL_PROBLEM_H

#include <memory>
#include <vector>
#include <dolfin/common/Hierarchical.h>

namespace dolfin
{

  // Forward declarations
  class Form;
  class Function;
  class FunctionSpace;
  class DirichletBC;
  class GenericVector;

  /// This class represents a nonlinear variational problem:
  ///
  /// Find u in V such that
  ///
  ///     F(u; v) = 0  for all v in V^,
  ///
  /// where V is the trial space and V^ is the test space.

  class NonlinearVariationalProblem
    : public Hierarchical<NonlinearVariationalProblem>
  {
  public:

    /// Create nonlinear variational problem, shared pointer version.
    /// The Jacobian form is specified which allows the use of a
    /// nonlinear solver that relies on the Jacobian (using Newton's
    /// method).
    NonlinearVariationalProblem(std::shared_ptr<const Form> F,
                                std::shared_ptr<Function> u,
                                std::vector<std::shared_ptr<const DirichletBC>> bcs,
                                std::shared_ptr<const Form> J=nullptr);

    /// Set the bounds for bound constrained solver
    void set_bounds(const Function& lb_func, const Function& ub_func);

    /// Set the bounds for bound constrained solver
    void set_bounds(std::shared_ptr<const GenericVector> lb,
                    std::shared_ptr<const GenericVector> ub);

    /// Return residual form
    std::shared_ptr<const Form> residual_form() const;

    /// Return Jacobian form
    std::shared_ptr<const Form> jacobian_form() const;

    /// Return solution variable
    std::shared_ptr<Function> solution();

    /// Return solution variable (const version)
    std::shared_ptr<const Function> solution() const;

    /// Return boundary conditions
    std::vector<std::shared_ptr<const DirichletBC>> bcs() const;

    /// Return trial space
    std::shared_ptr<const FunctionSpace> trial_space() const;

    /// Return test space
    std::shared_ptr<const FunctionSpace> test_space() const;

    /// Return lower bound
    std::shared_ptr<const GenericVector> lower_bound() const;

    /// Return upper bound
    std::shared_ptr<const GenericVector> upper_bound() const;

    /// Check whether Jacobian has been defined
    bool has_jacobian() const;

    /// Check whether lower bound has been defined
    bool has_lower_bound() const;

    /// Check whether upper bound have has defined
    bool has_upper_bound() const;

  private:

    // Check forms
    void check_forms() const;

    // The residual form
    std::shared_ptr<const Form> _residual;

    // The Jacobian form (pointer may be null if not provided)
    std::shared_ptr<const Form> _jacobian;

    // The solution
    std::shared_ptr<Function> _u;

    // The boundary conditions
    std::vector<std::shared_ptr<const DirichletBC>> _bcs;

    // The lower and upper bounds (pointers may be null if not
    // provided)
    std::shared_ptr<const GenericVector> _lb;
    std::shared_ptr<const GenericVector> _ub;
  };

}

#endif
