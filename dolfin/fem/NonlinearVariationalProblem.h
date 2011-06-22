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
// First added:  2011-06-22
// Last changed: 2011-06-22

#ifndef __NONLINEAR_VARIATIONAL_PROBLEM_H
#define __NONLINEAR_VARIATIONAL_PROBLEM_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/Hierarchical.h>

namespace dolfin
{

  // Forward declarations
  class Form;
  class Function;
  class FunctionSpace;
  class BoundaryCondition;

  /// This class represents a nonlinear variational problem:
  ///
  /// Find u in V such that
  ///
  ///     F(u; v) = 0  for all v in V^,
  ///
  /// where V is the trial space and V^ is the test space.

  class NonlinearVariationalProblem : public Hierarchical<NonlinearVariationalProblem>
  {
  public:

    // Developer note: The rhs argument (which must be zero) is included
    // here for two reasons; first to make the interface consistent with
    // the interface of LinearVariationalProblem and second to allow all
    // checks of arguments to be performed in a single place (not also
    // in the Equation class).

    /// Create linear variational problem
    NonlinearVariationalProblem(const Form& F,
                                int rhs,
                                Function& u,
                                std::vector<const BoundaryCondition*> bcs);

    /// Create linear variational problem (shared pointer version)
    NonlinearVariationalProblem(boost::shared_ptr<const Form> F,
                                int rhs,
                                boost::shared_ptr<Function> u,
                                std::vector<boost::shared_ptr<const BoundaryCondition> > bcs);

    /// Return residual form
    boost::shared_ptr<const Form> residual_form() const;

    /// Return Jacobian form
    boost::shared_ptr<const Form> jacobian_form() const;

    /// Return solution variable
    boost::shared_ptr<Function> solution();

    /// Return solution variable (const version)
    boost::shared_ptr<const Function> solution() const;

    /// Return boundary conditions
    std::vector<boost::shared_ptr<const BoundaryCondition> > bcs() const;

    /// Return trial space
    boost::shared_ptr<const FunctionSpace> trial_space() const;

    /// Return test space
    boost::shared_ptr<const FunctionSpace> test_space() const;

    /// Set Jacobian
    void set_jacobian(const Form& J);

    /// Set Jacobian (shared pointer version)
    void set_jacobian(boost::shared_ptr<const Form> J);

    /// Check whether Jacobian has been defined
    bool has_jacobian() const;

  private:

    // Check forms
    void check_forms(int rhs) const;

    // The residual form
    boost::shared_ptr<const Form> _F;

    // The Jacobian form (pointer may be zero if not provided)
    boost::shared_ptr<const Form> _J;

    // The solution
    boost::shared_ptr<Function> _u;

    // The boundary conditions
    std::vector<boost::shared_ptr<const BoundaryCondition> > _bcs;

  };

}

#endif
