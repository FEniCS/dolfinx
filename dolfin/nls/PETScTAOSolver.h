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

#include "PETScSNESSolver.h"
#include <petsctao.h>

namespace dolfin
{

  /// Forward declarations
  class PETScVector;

  /// This class implements methods for solving nonlinear optimisation
  /// problems via PETSc's TAO interface. It supports unconstrained
  /// minimisation as well as bound-constrained optimisation problem

  class PETScTAOSolver : public PETScObject
  {
  public:

    /// Create TAO solver for a particular method
    PETScTAOSolver(std::string method="default");

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

    /// Return a list of available solver methods
    // static std::vector<std::pair<std::string, std::string> > methods();

    /// Set the TAO solver type
    void set_solver(const std::string& method);

    /// Default parameter values
    static Parameters default_parameters();

    Parameters parameters;

    /// Return PETSc TAO pointer
    Tao tao() const
    { return _tao; }

  private:
    /// TAO context
    struct tao_ctx_t
    {
      OptimisationProblem* optimisation_problem;
      PETScVector* x;
    };
    struct tao_ctx_t _tao_ctx;

    /// PETSc solver pointer
    Tao _tao;

    /// Initialize TAO solver
    void init(const std::string& method);

    /// Compute the nonlinear objective function f(x) as well as
    /// its gradient g(x) = f'(x)
    static PetscErrorCode FormFunctionGradient(Tao tao, Vec x,
                                               PetscReal *fobj, Vec G,
                                               void *ctx);

    // Compute the hessian H(x) = f''(x)
    static PetscErrorCode FormHessian(Tao tao, Vec x, Mat H, Mat Hpre,
                                      void *ctx);

    // The Hessian matrix
    std::shared_ptr<const PETScMatrix> _H;
    
  };

}

#endif

#endif
