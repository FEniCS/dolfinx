// Copyright (C) 2008-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Marie E. Rognes 2011
//
// First added:  2008-12-26
// Last changed: 2011-01-07

#ifndef __VARIATIONAL_PROBLEM_H
#define __VARIATIONAL_PROBLEM_H

#include <vector>
#include <boost/scoped_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/adaptivity/AdaptiveSolver.h>

namespace dolfin
{

  class Form;
  class BoundaryCondition;
  class Function;
  class GoalFunctional;
  class ErrorControl;

  /// This class represents a (system of) partial differential
  /// equation(s) in variational form: Find u in V such that
  ///
  ///     F_u(v) = 0  for all v in V'.
  ///
  /// The variational problem is defined in terms of a bilinear
  /// form a(v, u) and a linear for L(v).
  ///
  /// For a linear variational problem, F_u(v) = a(v, u) - L(v),
  /// the forms should correspond to the canonical formulation
  ///
  ///     a(v, u) = L(v)  for all v in V'.
  ///
  /// For a nonlinear variational problem, the forms should
  /// be given by
  ///
  ///     a(v, u) = F_u'(v) u = F_u'(v, u),
  ///     L(v)    = F(v),
  ///
  /// that is, a(v, u) should be the Frechet derivative of F_u
  /// with respect to u, and L = F.
  ///
  /// Parameters:
  ///
  ///     "linear solvers": "direct" or "iterative" (default: "direct")
  ///     "symmetric":      true or false (default: false)

  class VariationalProblem : public Variable, public NonlinearProblem
  {
  public:

    /// Define variational problem with natural boundary conditions
    VariationalProblem(const Form& F, const Form& jacobian);

    /// Define variational problem with a single Dirichlet boundary conditions
    VariationalProblem(const Form& F,
                       const Form& jacobian,
                       const BoundaryCondition& bc);

    /// Define variational problem with a list of Dirichlet boundary conditions
    VariationalProblem(const Form& F,
                       const Form& jacobian,
                       const std::vector<const BoundaryCondition*>& bcs);


    /// Define variational problem with a list of Dirichlet boundary conditions
    /// and subdomains
    VariationalProblem(const Form& F,
                       const Form& jacobian,
                       const std::vector<const BoundaryCondition*>& bcs,
                       const MeshFunction<uint>* cell_domains,
                       const MeshFunction<uint>* exterior_facet_domains,
                       const MeshFunction<uint>* interior_facet_domains);

    /// Destructor
    ~VariationalProblem();

    /// is_linear
    const bool is_nonlinear() const {
      return nonlinear;
    };

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

    // Friends can access private variables.
    friend class AdaptiveSolver;

  private:

    // Initialize private forms based on checking of input
    const Form& init_rhs(const Form& F, const Form& jacobian);
    const Form& init_lhs(const Form& F, const Form& jacobian);
    bool is_nonlinear(const Form &F);

    // Solve linear variational problem
    void solve_linear(Function& u);

    // Solve nonlinear variational problem
    void solve_nonlinear(Function& u);

    // Bilinear form
    const Form& a;

    // Linear form
    const Form& L;

    // Boundary conditions
    std::vector<const BoundaryCondition*> bcs;

    // Mesh functions for assembly
    const MeshFunction<uint>* cell_domains;
    const MeshFunction<uint>* exterior_facet_domains;
    const MeshFunction<uint>* interior_facet_domains;

    // True if problem is nonlinear
    bool nonlinear;

    // Indicates whether the Jacobian matrix has been initialised
    bool jacobian_initialised;

    // Newton solver
    boost::scoped_ptr<NewtonSolver> _newton_solver;
  };

}

#endif
