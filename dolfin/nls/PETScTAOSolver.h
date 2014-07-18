// Copyright (C) 2014 Tianyi Li
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
// First added:  2014-06-22
// Last changed: 2014-06-22

#ifndef __PETSC_TAO_SOLVER_H
#define __PETSC_TAO_SOLVER_H

#ifdef ENABLE_PETSC_TAO

#include <map>
#include <petsctao.h>
#include <memory>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/la/PETScObject.h>
#include <dolfin/la/PETScVector.h>

namespace dolfin
{

  /// Forward declarations
  class GenericVector;
  // class PETScMatrix;
  class PETScVector;
  class OptimisationProblem;

  /// This class implements methods for solving nonlinear optimisation
  /// problems via PETSc's TAO interface. It supports unconstrained
  /// minimisation as well as bound-constrained optimisation problem

  class PETScTAOSolver : public PETScObject
  {
  public:

    /// Create TAO solver for a particular method
    PETScTAOSolver(const std::string tao_type="default",
                   const std::string ksp_type="default",
                   const std::string pc_type="default");

    /// Destructor
    virtual ~PETScTAOSolver();

    /// Solve a nonlinear bound-constrained optimisation problem
  
    /// *Arguments*
    ///     optimisation_problem (:py:class:`OptimisationProblem`)
    ///         The nonlinear optimisation problem.
    ///     x (:py:class:`GenericVector`)
    ///         The vector.
    ///     lb (:py:class:`GenericVector`)
    ///         The lower bound.
    ///     ub (:py:class:`GenericVector`)
    ///         The upper bound.
    
    /// *Returns*
    ///     number of iterations
    std::size_t solve(OptimisationProblem& optimisation_problem,
                      GenericVector& x,
                      const GenericVector& lb,
                      const GenericVector& ub);

    /// Solve a nonlinear bound-constrained optimisation problem
  
    /// *Arguments*
    ///     optimisation_problem (:py:class:`OptimisationProblem`)
    ///         The nonlinear optimisation problem.
    ///     x (:py:class:`PETScVector`)
    ///         The vector.
    ///     lb (:py:class:`PETScVector`)
    ///         The lower bound.
    ///     ub (:py:class:`PETScVector`)
    ///         The upper bound.
    
    /// *Returns*
    ///     number of iterations
    std::size_t solve(OptimisationProblem& optimisation_problem,
                      PETScVector& x,
                      const PETScVector& lb,
                      const PETScVector& ub);

    /// Return a list of available solver methods
    // static std::vector<std::pair<std::string, std::string> > methods();

    /// Set the TAO solver type
    void set_solver(const std::string tao_type="default");

    /// Set PETSc Krylov Solver (KSP) used by TAO
    void set_ksp_pc(const std::string ksp_type="default",
                    const std::string pc_type="default");

    /// Return a list of available solver methods
    static std::vector<std::pair<std::string, std::string> > methods();

    /// Default parameter values
    static Parameters default_parameters();

    Parameters parameters;

    /// Return PETSc TAO pointer
    Tao tao() const
    { return _tao; }

  private:
    /// TAO context for optimisation problems
    struct tao_ctx_t
    {
      OptimisationProblem* optimisation_problem;
      PETScVector* x;
    };
    struct tao_ctx_t _tao_ctx;

    /// PETSc solver pointer
    Tao _tao;

    /// Hessian matrix
    // PETScMatrix _H;

    /// Set options
    void set_options();
    void set_ksp_options();

    /// Available solvers
    static const std::map<std::string, std::pair<std::string, const TaoType> > _methods;

    /// Initialize TAO solver
    void init(const std::string tao_type="default",
              const std::string ksp_type="default",
              const std::string pc_type="default");

    /// Compute the nonlinear objective function f(x) as well as
    /// its gradient g(x) = f'(x)
    static PetscErrorCode FormFunctionGradient(Tao tao, Vec x,
                                               PetscReal *fobj, Vec G,
                                               void *ctx);

    /// Compute the hessian H(x) = f''(x)
    static PetscErrorCode FormHessian(Tao tao, Vec x, Mat H, Mat Hpre,
                                      void *ctx);
    
  };

}

#endif

#endif
