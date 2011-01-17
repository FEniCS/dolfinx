// Copyright (C) 2008-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Marie E. Rognes 2011
//
// First added:  2008-12-26
// Last changed: 2011-01-17

#ifndef __VARIATIONAL_PROBLEM_H
#define __VARIATIONAL_PROBLEM_H

#include <dolfin/common/Variable.h>
#include <dolfin/adaptivity/AdaptiveSolver.h>
#include "NonlinearVariationalSolver.h"
#include "LinearVariationalSolver.h"

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
  /// _Form_s and, optionally, a set of _BoundaryCondition_s and
  /// _MeshFunction_s that specify any subdomains involved in the
  /// definition of the _Form_s.
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

  class VariationalProblem : public Variable
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
    /// and subdomains for cells, exterior and interior facets of the mesh
    VariationalProblem(const Form& form_0,
                       const Form& form_1,
                       const std::vector<const BoundaryCondition*>& bcs,
                       const MeshFunction<uint>* cell_domains,
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains);

    /// Destructor
    ~VariationalProblem();

    /// Solve variational problem
    void solve(Function& u);

    /// Solve variational problem and extract sub functions
    void solve(Function& u0, Function& u1);

    /// Solve variational problem and extract sub functions
    void solve(Function& u0, Function& u1, Function& u2);

    /// Solve variational problem adaptively to within given tolerance
    void solve(Function& u, double tol, GoalFunctional& M);

    /// Solve variational problem adaptively to within given tolerance
    void solve(Function& u, double tol, Form& M, ErrorControl& ec);

    /// Return true if problem is non-linear
    const bool is_nonlinear() const;

    /// Return trial space for variational problem
    const FunctionSpace& trial_space() const;

    /// Return test space for variational problem
    const FunctionSpace& test_space() const;

    /// Return the bilinear form
    const Form& bilinear_form() const;

    /// Return the linear form
    const Form& linear_form() const;

    /// Return the list of boundary conditions
    const std::vector<const BoundaryCondition*> bcs() const;

    /// Return the cell domains
    const MeshFunction<uint>* cell_domains() const;

    /// Return the exterior facet domains
    const MeshFunction<uint>* exterior_facet_domains() const;

    /// Return the interior facet domains
    const MeshFunction<uint>* interior_facet_domains() const;

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

    // Extract which of the two forms is linear
    static const Form& extract_linear_form(const Form& form_0,
                                           const Form& form_1);

    // Extract which of the two forms is bilinear
    static const Form& extract_bilinear_form(const Form& form_0,
                                             const Form& form_1);

    // Initialize parameters
    void init_parameters();

    // Print error message when form arguments are incorrect
    static void form_error();

    // True if problem is nonlinear
    bool _is_nonlinear;

    // Linear form
    const Form& _linear_form;

    // Bilinear form
    const Form& _bilinear_form;

    // Boundary conditions
    std::vector<const BoundaryCondition*> _bcs;

    // Mesh functions for assembly
    const MeshFunction<uint>* _cell_domains;
    const MeshFunction<uint>* _exterior_facet_domains;
    const MeshFunction<uint>* _interior_facet_domains;

  };

}

#endif
