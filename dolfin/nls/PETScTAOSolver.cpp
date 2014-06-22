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

#ifdef ENABLE_PETSC_TAO

#include <map>
#include <string>
#include <utility>
#include <petscsys.h>
#include <boost/assign/list_of.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/timing.h>
#include <dolfin/common/Timer.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScLUSolver.h>
#include <dolfin/la/PETScPreconditioner.h>
#include <dolfin/la/PETScVector.h>
#include "dolfin/la/PETScKrylovSolver.h"
#include "OptimisationProblem.h"
#include "PETScTAOSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters PETScTAOSolver::default_parameters()
{
  Parameters p("tao_solver");

  p.add("monitor_convergence"    ,false);
  p.add("report"                 ,false);
  p.add("function_absolute_tol"  ,1.0e-10);
  p.add("function_relative_tol"  ,1.0e-10);
  p.add("gradient_absolute_tol"  ,1.0e-08);
  p.add("gradient_relative_tol"  ,1.0e-08);
  p.add("gradient_t_tol"         ,0.0);
  p.add("error_on_nonconvergence",true);
  p.add("maximum_iterations"     ,100);
  p.add("options_prefix"         ,"default");

  Parameters ksp("krylov_solver");
  ksp = KrylovSolver::default_parameters();
  p.add(ksp);

  return p;
}
//-----------------------------------------------------------------------------
PETScTAOSolver::PETScTAOSolver(std::string method) : _tao(NULL)
{
  // Check that the requested method is known
  if (_methods.count(nls_type) == 0)
  {
    dolfin_error("PETScTAOSolver.cpp",
                 "create PETSc TAO solver",
                 "Unknown TAO method \"%s\"", nls_type.c_str());
  }

  // Set parameter values
  parameters = default_parameters();

  //Initialize the TAO solver
  init(method);
}
//-----------------------------------------------------------------------------
PETScTAOSolver::~PETScTAOSolver()
{
  if (_tao)
    TaoDestroy(&_tao);
}
//-----------------------------------------------------------------------------
void PETScTAOSolver::init(const std::string& method)
{
  if (_tao)
    TaoDestroy(&_tao);

  // Create TAO object
  TaoCreate(PETSC_COMM_WORLD, &_tao);

  // Set tao solver
  set_solver(method);
}
//-----------------------------------------------------------------------------
std::size_t PETScTAOSolver::solve(OptimisationProblem& optimisation_problem,
                                  GenericVector& x,
                                  const GenericVector& lb,
                                  const GenericVector& ub)
{
  // Set initial vector
  TaoSetInitialVector(_tao, x.down_cast<PETScVector>().vec());

  // Set the bound on the variables
  TaoSetVariableBounds(_tao,
                       lb.down_cast<PETScVector>().vec(),
                       ub.down_cast<PETScVector>().vec());

  // Set the user function, gradient and Hessian evaluation routines and
  // data structures
  TaoSetObjectiveAndGradientRoutine(_tao,
                                    FormFunctionGradient,
                                    this);
  TaoSetHessianRoutine(_tao, _H->mat(), _H->mat(),
                       FormHessian, this);

  // Solve the bound constrained problem
  Timer timer("TAO solver");
  const char* tao_type;
  TaoGetType(_tao, &tao_type);
  log(PROGRESS, "TAO solver %s starting to solve %i x %i system", tao_type,
      A->size(0), A->size(1));

  // Solve
  TaoSolve(_tao);

  // Update ghost values
  x.update_ghost_values();

  // Print the report on convergences and methods used
  if (parameters["report"])
    TaoView(_tao, PETSC_VIEWER_STDOUT_WORLD);

  // Check for convergence
  TaoConvergedReason reason;
  TaoGetConvergedReason(_tao, &reason);

  // Get the number of iterations
  PetscInt num_iterations = 0;
  TaoGetMaximumIterations(_tao, &num_iterations);

  // Report number of iterations
  if (reason >= 0)
    log(PROGRESS, "Tao solver converged\n");
  else
  {
    bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
    if (error_on_nonconvergence)
    {
      TaoView(_tao, PETSC_VIEWER_STDOUT_WORLD);
      dolfin_error("PETScTAOSolver.cpp",
                   "solve nonlinear optimisation problem",
                   "Solution failed to converge in %i iterations (TAO reason %d)",
                   num_iterations, reason);
    }
    else
    {
      log(WARNING,  "TAO solver %s failed to converge. Try a different TAO method" \
                    " or adjust some parameters.", tao_type);
    }
  }

  return num_iterations;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScTAOSolver::FormFunctionGradient(Tao tao, Vec x,
                                                    PetscReal *fobj, Vec G,
                                                    void *ctx)
{
  // Get the optimisation problem object
  struct tao_ctx_t tao_ctx = *(struct tao_ctx_t*) ctx;
  OptimisationProblem* optimisation_problem = tao_ctx.optimisation_problem;

  // Wrap the PETSc objects
  PETScVector x_wrap(x);
  PETScVector G_wrap(G);

  // Compute the objective function f(x) and its gradient G(x) = f'(x)
  PETScMatrix H;
  optimisation_problem->f(*fobj, x_wrap)
  optimisation_problem->form(H, G_wrap, x_wrap);
  optimisation_problem->F(G_wrap, x_wrap);

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScTAOSolver::FormHessian(Tao tao, Vec x, Mat H, Mat Hpre,
                                           void *ctx)
{
  // Get the optimisation problem object
  struct tao_ctx_t tao_ctx = *(struct tao_ctx_t*) ctx;
  OptimisationProblem* optimisation_problem = tao_ctx.optimisation_problem;

  // Wrap the PETSc objects
  PETScVector x_wrap(x);
  PETScVector H_wrap(H);

  // Compute the hessian H(x) = f''(x)
  PETScVector G;
  optimisation_problem->form(H_wrap, G, x_wrap);
  optimisation_problem->J(H_wrap, x_wrap);

  return 0;
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_solver(const std::string& method)
{
  dolfin_assert(_tao);

  // Do nothing if default type is specified
  if (method == "default")
    TaoSetType(_tao, "tron");
  else
  {
    // Choose solver
    if (method == "tron")
      TaoSetType(_tao, "tron");
    else if (method == "blmvm")
      TaoSetType(_tao, "blmvm" );
    else if (method == "gpcg")
      TaoSetType(_tao, "gpcg" );
    else if (method == "bqpip")
      TaoSetType(_tao, "bqpip");
    else
    {
      dolfin_error("TAOLinearBoundSolver.cpp",
       "set solver for TAO solver",
                   "Unknown solver type (\"%s\")", method.c_str());
    }
  }
}
//-----------------------------------------------------------------------------

#endif
