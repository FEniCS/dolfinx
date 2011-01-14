// Copyright (C) 2008-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Marie E. Rognes 2011
//
// First added:  2008-12-26
// Last changed: 2011-01-14

#ifndef __VARIATIONAL_PROBLEM_H
#define __VARIATIONAL_PROBLEM_H

#include <vector>
#include <boost/scoped_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/adaptivity/AdaptiveSolver.h>

namespace dolfin
{

  class Form;
  class BoundaryCondition;
  class Function;
  class NewtonSolver;
  class GoalFunctional;
  class ErrorControl;

  class FunctionSpace;

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

  class VariationalProblem : public Variable, public NonlinearProblem
  {
  public:

    /// Define variational problem with natural boundary conditions
    VariationalProblem(const Form& a, const Form& L);

    /// Define variational problem with a single Dirichlet boundary conditions
    VariationalProblem(const Form& a, const Form& L,
                       const BoundaryCondition& bc);

    /// Define variational problem with a list of Dirichlet boundary conditions
    VariationalProblem(const Form& a, const Form& L,
                       const std::vector<const BoundaryCondition*>& bcs);

    /// Define variational problem with a list of Dirichlet boundary conditions
    /// and subdomains
    VariationalProblem(const Form& a,
                       const Form& L,
                       const std::vector<const BoundaryCondition*>& bcs,
                       const MeshFunction<uint>* cell_domains,
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains);

    /// Destructor
    ~VariationalProblem();

    /// Return true if problem is non-linear
    const bool is_nonlinear() const;

    /// Solve variational problem
    void solve(Function& u);

    /// Solve variational problem adaptively to given tolerance
    void solve(Function& u, const double tol, GoalFunctional& M);

    /// Solve variational problem adaptively to given tolerance with
    /// given error controller
    void solve(Function& u, const double tol, Form& M, ErrorControl& ec);

    /// Solve variational problem and extract sub functions
    void solve(Function& u0, Function& u1);

    /// Solve variational problem and extract sub functions
    void solve(Function& u0, Function& u1, Function& u2);

    /// Compute F at current point x
    void F(GenericVector& b, const GenericVector& x);

    /// Compute J = F' at current point x
    void J(GenericMatrix& A, const GenericVector& x);

    /// Optional callback called before calls to F() and J()
    virtual void update(const GenericVector& x);

    /// Return Newton solver (only useful when solving a nonlinear problem)
    NewtonSolver& newton_solver();

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("variational_problem");

      p.add("linear_solver",  "lu");
      p.add("preconditioner", "default");
      p.add("symmetric", false);
      p.add("reset_jacobian", true);

      p.add("print_rhs", false);
      p.add("print_matrix", false);

      p.add(NewtonSolver::default_parameters());
      p.add(LUSolver::default_parameters());
      p.add(KrylovSolver::default_parameters());
      p.add(AdaptiveSolver::default_parameters());

      return p;
    }

    friend class AdaptiveSolver;

    // FIXME: New functions below to be implemented
    const FunctionSpace& trial_space() const;
    const Form& bilinear_form() const;
    const Form& linear_form() const;
    const std::vector<const BoundaryCondition*> bcs() const;
    const MeshFunction<uint>* cell_domains() const;
    const MeshFunction<uint>* exterior_facet_domains() const;
    const MeshFunction<uint>* interior_facet_domains() const;

  private:

    // Solve linear variational problem
    void solve_linear(Function& u);

    // Solve nonlinear variational problem
    void solve_nonlinear(Function& u);

    // Extract bilinear and linear forms
    const Form& extract_bilinear(const Form& b, const Form& c) const;
    const Form& extract_linear(const Form& b, const Form& c) const;

    // Detect whether problem is nonlinear
    bool is_nonlinear(const Form &b, const Form& c) const;

    // Forms (old names, will be deleted)
    const Form& a;

    // Linear form
    const Form& L;

    // Bilinear form
    //const Form& _bilinear_form;

    // Linear form
    //const Form& _linear_form;

    // Boundary conditions
    std::vector<const BoundaryCondition*> _bcs;

    // Mesh functions for assembly
    const MeshFunction<uint>* _cell_domains;
    const MeshFunction<uint>* _exterior_facet_domains;
    const MeshFunction<uint>* _interior_facet_domains;

    // True if problem is nonlinear
    bool nonlinear;

    // Indicates whether the Jacobian matrix has been initialised
    bool jacobian_initialised;

    // Newton solver
    boost::scoped_ptr<NewtonSolver> _newton_solver;
  };

}

#endif
