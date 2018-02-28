// Copyright (C) 2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <vector>

namespace dolfin
{

// Forward declarations
class PETScVector;

namespace function
{
class Function;
class FunctionSpace;
}

namespace fem
{
class DirichletBC;
class Form;

/// This class represents a nonlinear variational problem:
///
/// Find u in V such that
///
///     F(u; v) = 0  for all v in V^,
///
/// where V is the trial space and V^ is the test space.

class NonlinearVariationalProblem
{
public:
  /// Create nonlinear variational problem, shared pointer version.
  /// The Jacobian form is specified which allows the use of a
  /// nonlinear solver that relies on the Jacobian (using Newton's
  /// method).
  NonlinearVariationalProblem(
      std::shared_ptr<const Form> F, std::shared_ptr<function::Function> u,
      std::vector<std::shared_ptr<const fem::DirichletBC>> bcs,
      std::shared_ptr<const Form> J = nullptr);

  /// Set the bounds for bound constrained solver
  void set_bounds(const function::Function& lb_func,
                  const function::Function& ub_func);

  /// Set the bounds for bound constrained solver
  void set_bounds(std::shared_ptr<const PETScVector> lb,
                  std::shared_ptr<const PETScVector> ub);

  /// Return residual form
  std::shared_ptr<const Form> residual_form() const;

  /// Return Jacobian form
  std::shared_ptr<const Form> jacobian_form() const;

  /// Return solution variable
  std::shared_ptr<function::Function> solution();

  /// Return solution variable (const version)
  std::shared_ptr<const function::Function> solution() const;

  /// Return boundary conditions
  std::vector<std::shared_ptr<const fem::DirichletBC>> bcs() const;

  /// Return trial space
  std::shared_ptr<const function::FunctionSpace> trial_space() const;

  /// Return test space
  std::shared_ptr<const function::FunctionSpace> test_space() const;

  /// Return lower bound
  std::shared_ptr<const PETScVector> lower_bound() const;

  /// Return upper bound
  std::shared_ptr<const PETScVector> upper_bound() const;

private:
  // Check forms
  void check_forms() const;

  // The residual form
  std::shared_ptr<const Form> _residual;

  // The Jacobian form (pointer may be null if not provided)
  std::shared_ptr<const Form> _jacobian;

  // The solution
  std::shared_ptr<function::Function> _u;

  // The boundary conditions
  std::vector<std::shared_ptr<const fem::DirichletBC>> _bcs;

  // The lower and upper bounds (pointers may be null if not
  // provided)
  std::shared_ptr<const PETScVector> _lb;
  std::shared_ptr<const PETScVector> _ub;
};
}
}