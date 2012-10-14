// Copyright (C) 2012 Patrick E. Farrell
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
//
// First added:  2012-10-13
// Last changed: 2012-10-13

#ifndef __PETSC_SNES_SOLVER_H
#define __PETSC_SNES_SOLVER_H

#ifdef HAS_PETSC

#include <map>
#include <petscsnes.h>
#include <boost/shared_ptr.hpp>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/nls/NewtonSolver.h>

namespace dolfin
{

  /// This class implements methods for solving nonlinear systems
  /// via PETSc's SNES interface. It includes line search and trust
  /// region techniques for globalising the convergence of the
  /// nonlinear iteration.
  class PETScSNESSolver
  {
  public:

    /// Create SNES solver for a particular method
    PETScSNESSolver(std::string nls_type="default");

    /// Destructor
    virtual ~PETScSNESSolver();

    /// Solve abstract nonlinear problem :math:`F(x) = 0` for given
    /// :math:`F` and Jacobian :math:`\dfrac{\partial F}{\partial x}`.
    ///
    /// *Arguments*
    ///     nonlinear_function (_NonlinearProblem_)
    ///         The nonlinear problem.
    ///     x (_GenericVector_)
    ///         The vector.
    ///
    /// *Returns*
    ///     std::pair<uint, bool>
    ///         Pair of number of Newton iterations, and whether
    ///         iteration converged)
    std::pair<uint, bool> solve(NonlinearProblem& nonlinear_function,
                                GenericVector& x);

    /// Return a list of available solver methods
    static std::vector<std::pair<std::string, std::string> > methods();

    /// Default parameter values
    static Parameters default_parameters();

    Parameters parameters;

  private:

    /// PETSc solver pointer
    boost::shared_ptr<SNES> _snes;

    /// Initialize SNES solver
    void init(const std::string& method);

    /// Available solvers
    static const std::map<std::string, const SNESType> _methods;

    /// Available solvers descriptions
    static const std::vector<std::pair<std::string, std::string> > _methods_descr;

    /// The callback for PETSc to compute F
    static PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void* ctx);

    /// The callback for PETSc to compute A
    static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat* A, Mat* B, MatStructure* flag, void* ctx);
  };

}

#endif

#endif
