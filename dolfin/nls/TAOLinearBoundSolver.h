// Copyright (C) 2012 Corrado Maurini
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

#ifndef _TAOLinearBoundSolver_H
#define _TAOLinearBoundSolver_H

#ifdef HAS_PETSC

#include <map>
#include <memory>
#include <petscksp.h>
#include <petscpc.h>

#include <petsctao.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/types.h>

#include <dolfin/la/PETScObject.h>
#include <dolfin/la/KrylovSolver.h>

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScMatrix;
  class PETScVector;
  class PETScPreconditioner;
  class PETScKrylovSolver;
  class PETScKSPDeleter;

  /// This class provides a bound constrained solver for a
  /// linear variational inequality defined by a matrix A and a vector b.
  /// It solves the problem:
  ///
  /// Find :math:`x_l\leq x\leq x_u` such that
  /// :math:`(Ax-b)\cdot (y-x)\geq 0,\; \forall x_l\leq y\leq x_u`
  ///
  /// It is a wrapper for the TAO bound constrained solver.
  ///
  /// *Example*
  ///    .. code-block:: python
  ///
  ///       # Assemble the linear system
  ///       A, b = assemble_system(a, L, bc)
  ///       # Define the constraints
  ///       constraint_u = Constant(1.)
  ///       constraint_l = Constant(0.)
  ///       u_min = interpolate(constraint_l, V)
  ///       u_max = interpolate(constraint_u, V)
  ///       # Define the function to store the solution
  ///       usol=Function(V)
  ///       # Create the TAOLinearBoundSolver
  ///       solver=TAOLinearBoundSolver("tao_gpcg","gmres")
  ///       # Set some parameters
  ///       solver.parameters["monitor_convergence"]=True
  ///       solver.parameters["report"]=True
  ///       # Solve the problem
  ///       solver.solve(A, usol.vector(), b , u_min.vector(), u_max.vector())
  ///       info(solver.parameters,True)
  ///
  class TAOLinearBoundSolver : public Variable, public PETScObject
  {
  public:

    /// Create TAO bound constrained solver
    explicit TAOLinearBoundSolver(MPI_Comm comm);

    /// Create TAO bound constrained solver
    TAOLinearBoundSolver(const std::string method = "default",
                         const std::string ksp_type = "default",
                         const std::string pc_type = "default");

    /// Destructor
    ~TAOLinearBoundSolver();

    /// Solve the linear variational inequality defined by A and b
    /// with xl =< x <= xu
    std::size_t solve(const GenericMatrix& A, GenericVector& x,
                      const GenericVector& b, const GenericVector& xl,
                      const GenericVector& xu);

    /// Solve the linear variational inequality defined by A and b
    /// with xl =< x <= xu
    std::size_t solve(const PETScMatrix& A, PETScVector& x,
                      const PETScVector& b,
                      const PETScVector& xl, const PETScVector& xu);

    // Set the TAO solver type
    void set_solver(const std::string&);

    /// Set PETSC Krylov Solver (ksp) used by TAO
    void set_ksp(const std::string ksp_type = "default");

    // Return TAO solver pointer
    Tao tao() const;

    /// Return a list of available Tao solver methods
    static std::map<std::string, std::string> methods();

    /// Return a list of available krylov solvers
    static std::map<std::string, std::string> krylov_solvers();

    /// Return a list of available preconditioners
    static std::map<std::string, std::string> preconditioners();

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("tao_solver");

      p.add("monitor_convergence"    , false);
      p.add("report"                 , false);
      p.add("function_absolute_tol"  , 1.0e-10);
      p.add("function_relative_tol"  , 1.0e-10);
      p.add("gradient_absolute_tol"  , 1.0e-8);
      p.add("gradient_relative_tol"  , 1.0e-8);
      p.add("gradient_t_tol"         , 0.0);
      p.add("error_on_nonconvergence", true);
      p.add("maximum_iterations"     , 100);
      p.add("options_prefix"         , "default");

      Parameters ksp("krylov_solver");
      ksp = KrylovSolver::default_parameters();
      p.add(ksp);

      return p;
    }

    // Return Matrix shared pointer
    std::shared_ptr<const PETScMatrix> get_matrix() const;

    // Return load vector shared pointer
    std::shared_ptr<const PETScVector> get_vector() const;

  private:

    // Set operators with GenericMatrix and GenericVector
    void set_operators(std::shared_ptr<const GenericMatrix> A,
                       std::shared_ptr<const GenericVector> b);

    // Set operators with shared pointer to PETSc objects
    void set_operators(std::shared_ptr<const PETScMatrix> A,
                       std::shared_ptr<const PETScVector> b);

    // Callback for changes in parameter values
    void read_parameters();

    // Available ksp solvers
    static const std::map<std::string, const KSPType> _ksp_methods;

    // Available tao solvers descriptions
    static const std::map<std::string, std::string> _methods_descr;

    // Set options
    void set_ksp_options();

    // Initialize TAO solver
    void init(const std::string& method);

    // Tao solver pointer
    Tao _tao;

    // Petsc preconditioner
    std::shared_ptr<PETScPreconditioner> _preconditioner;

    // Operator (the matrix) and the vector
    std::shared_ptr<const PETScMatrix> _matA;
    std::shared_ptr<const PETScVector> _b;

    bool _preconditioner_set;

    // Computes the value of the objective function and its gradient.
    static PetscErrorCode
      __TAOFormFunctionGradientQuadraticProblem(Tao tao, Vec X,
                                                PetscReal *ener, Vec G,
                                                void *ptr);

    // Computes the hessian of the quadratic objective function
    static PetscErrorCode
      __TAOFormHessianQuadraticProblem(Tao tao,Vec X, Mat H, Mat Hpre,
                                       void *ptr);

    //-------------------------------------------------------------------------
    //  Monitor the state of the solution at each iteration. The
    //  output printed to the screen is:
    //
    //  iterate - the current iterate number (>=0)
    //  f       - the current function value
    //  gnorm   - the square of the gradient norm, duality gap, or other
    //             measure
    //            indicating distance from optimality.
    //  cnorm - the infeasibility of the current solution with regard
    //         to the constraints.
    //  xdiff - the step length or trust region radius of the most
    //         recent iterate.
    //-------------------------------------------------------------------------
    static PetscErrorCode __TAOMonitor(Tao tao, void *ctx);

  };

}

#endif
#endif
