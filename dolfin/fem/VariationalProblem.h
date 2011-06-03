// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells
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
// Modified by Marie E. Rognes 2011
//
// First added:  2008-12-26
// Last changed: 2011-03-11

#ifndef __VARIATIONAL_PROBLEM_H
#define __VARIATIONAL_PROBLEM_H

#include <boost/shared_ptr.hpp>

#include <dolfin/common/Hierarchical.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class BoundaryCondition;
  class ErrorControl;
  class Form;
  class Function;
  class FunctionSpace;
  class GoalFunctional;
  template<class T> class MeshFunction;

  /// A _VariationalProblem_ represents a (system of) partial
  /// differential equation(s) in variational form:
  ///
  /// Find u_h in V_h such that
  ///
  ///     F(u_h; v) = 0  for all v in V_h',
  ///
  /// where V_h is the trial space and V_h' is the test space.
  ///
  /// The variational problem is specified in terms of a pair of
  /// _Form_s and, optionally, a set of _BoundaryCondition_s.
  ///
  /// The pair of forms may either specify a nonlinear problem
  ///
  ///    (1) F(u_h; v) = 0
  ///
  /// in terms of the residual F and its derivative J = F':
  ///
  ///    F, J  (F linear, J bilinear)
  ///
  /// or a linear problem
  ///
  ///    (2) F(u_h; v) = a(u_h, v) - L(v) = 0
  ///
  /// in terms of the bilinear form a and a linear form L:
  ///
  ///    a, L  (a bilinear, L linear)
  ///
  /// Thus, a pair of forms is interpreted either as a nonlinear
  /// problem or a linear problem depending on the ranks of the given
  /// forms.

  class VariationalProblem : public Variable, public Hierarchical<VariationalProblem>
  {
  public:

    /// Define variational problem with natural boundary conditions
    VariationalProblem(const Form& form_0,
                       const Form& form_1);

    /// Define variational problem with a single Dirichlet boundary condition
    VariationalProblem(const Form& form_0,
                       const Form& form_1,
                       const BoundaryCondition& bc);

    /// Define variational problem with a list of Dirichlet boundary conditions
    VariationalProblem(const Form& form_0,
                       const Form& form_1,
                       const std::vector<const BoundaryCondition*>& bcs);

    /// Define variational problem with a list of Dirichlet boundary conditions
    VariationalProblem(boost::shared_ptr<const Form> form_0,
                       boost::shared_ptr<const Form> form_1,
                       std::vector<boost::shared_ptr<const BoundaryCondition> > bcs);

    /// Destructor
    ~VariationalProblem();

    /// Solve variational problem
    void solve(Function& u) const;

    /// Solve variational problem and extract sub functions
    void solve(Function& u0, Function& u1) const;

    /// Solve variational problem and extract sub functions
    void solve(Function& u0, Function& u1, Function& u2) const;

    /// Solve variational problem adaptively to within given tolerance
    void solve(Function& u, const double tol, GoalFunctional& M) const;

    /// Solve variational problem adaptively to within given tolerance
    void solve(Function& u, const double tol, Form& M, ErrorControl& ec) const;

    /// Return true if problem is non-linear
    const bool is_nonlinear() const;

    /// Return trial space for variational problem
    const FunctionSpace& trial_space() const;

    /// Return test space for variational problem
    const FunctionSpace& test_space() const;

    /// Return the bilinear form
    boost::shared_ptr<const Form> bilinear_form() const;

    /// Return form_0
    boost::shared_ptr<const Form> form_0() const;

    /// Return form_1
    boost::shared_ptr<const Form> form_1() const;

    /// Return the linear form
    boost::shared_ptr<const Form> linear_form() const;

    /// Return the list of boundary conditions (shared_ptr version)
    const std::vector<boost::shared_ptr<const BoundaryCondition> > bcs() const;

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("variational_problem");
      p.add("symmetric", false);
      return p;
    }

  private:

    // Extract whether the problem is nonlinear
    static bool extract_is_nonlinear(const Form& form_0,
                                     const Form& form_1);

    // Extract whether the problem is nonlinear (shared_ptr version)
    static bool extract_is_nonlinear(boost::shared_ptr<const Form> form_0,
                                     boost::shared_ptr<const Form> form_1);

    // Extract which of the two forms is linear
    static boost::shared_ptr<const Form>
    extract_linear_form(const Form& form_0,
                        const Form& form_1);

    // Extract which of the two forms is linear (shared_ptr version)
    static boost::shared_ptr<const Form>
    extract_linear_form(boost::shared_ptr<const Form> form_0,
                        boost::shared_ptr<const Form> form_1);

    // Extract which of the two forms is bilinear
    static boost::shared_ptr<const Form>
    extract_bilinear_form(const Form& form_0,
                          const Form& form_1);

    // Extract which of the two forms is bilinear (shared_ptr version)
    static boost::shared_ptr<const Form>
    extract_bilinear_form(boost::shared_ptr<const Form> form_0,
                          boost::shared_ptr<const Form> form_1);

    // Initialize parameters
    void init_parameters();

    // Print error message when form arguments are incorrect
    static void form_error();

    // True if problem is nonlinear
    bool _is_nonlinear;

    // Linear form
    boost::shared_ptr<const Form> _linear_form;

    // Bilinear form
    boost::shared_ptr<const Form> _bilinear_form;

    // Boundary conditions
    std::vector<boost::shared_ptr<const BoundaryCondition> > _bcs;

  };

}

#endif
