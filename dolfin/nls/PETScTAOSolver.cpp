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
// Last changed: 2014-07-27

#ifdef HAS_PETSC

#include <map>
#include <string>
#include <utility>
#include <petscsys.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScLUSolver.h>
#include <dolfin/la/PETScPreconditioner.h>
#include <dolfin/la/PETScVector.h>
#include "OptimisationProblem.h"
#include "PETScTAOSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const std::map<std::string, std::pair<std::string, const TaoType>>
PETScTAOSolver::_methods
= { {"default", {"Default TAO method (ntl or tron)", TAOTRON}},
    {"tron",    {"Newton Trust Region method", TAOTRON}},
    {"bqpip",   {"Interior-Point Newton's method", TAOBQPIP}},
    {"gpcg",    {"Gradient projection conjugate gradient method", TAOGPCG}},
    {"blmvm",   {"Limited memory variable metric method", TAOBLMVM}},
    {"nls",     {"Newton's method with line search", TAONLS}},
    {"ntr",     {"Newton's method with trust region", TAONTR}},
    {"ntl",     {"Newton's method with trust region and line search", TAONTL}},
    {"cg",      {"Nonlinear conjugate gradient method", TAOCG}},
    {"nm",      {"Nelder-Mead algorithm", TAONM}} };
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string>> PETScTAOSolver::methods()
{
  std::vector<std::pair<std::string, std::string>> available_methods;
  for (auto it = _methods.begin(); it != _methods.end(); ++it)
    available_methods.push_back(std::make_pair(it->first, it->second.first));
  return available_methods;
}
//-----------------------------------------------------------------------------
Parameters PETScTAOSolver::default_parameters()
{
  Parameters p("tao_solver");

  p.add("monitor_convergence"    , false);
  p.add("report"                 , false);
  p.add("function_absolute_tol"  , 1.0e-10);
  p.add("function_relative_tol"  , 1.0e-10);
  p.add("gradient_absolute_tol"  , 1.0e-08);
  p.add("gradient_relative_tol"  , 1.0e-08);
  p.add("gradient_t_tol"         , 0.0);
  p.add("error_on_nonconvergence", true);
  p.add("maximum_iterations"     , 100);
  p.add("options_prefix"         , "default");
  p.add("method"                 , "default");
  p.add("linear_solver"          , "default");
  p.add("preconditioner"         , "default");

  std::set<std::string> line_searches;
  line_searches.insert("default");
  line_searches.insert("unit");
  line_searches.insert("more-thuente");
  line_searches.insert("gpcg");
  line_searches.insert("armijo");
  line_searches.insert("owarmijo");
  line_searches.insert("ipm");

  p.add("line_search", "default", line_searches);

  p.add(KrylovSolver::default_parameters());

  return p;
}
//-----------------------------------------------------------------------------
PETScTAOSolver::PETScTAOSolver(const std::string tao_type,
                               const std::string ksp_type,
                               const std::string pc_type) : _tao(NULL)
{
  // Create TAO object
  PetscErrorCode ierr = TaoCreate(PETSC_COMM_WORLD, &_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoCreate");

  // Set parameter values
  parameters = default_parameters();

  // Update parameters when tao/ksp/pc_types are explictly given
  update_parameters(tao_type, ksp_type, pc_type);
}
//-----------------------------------------------------------------------------
PETScTAOSolver::~PETScTAOSolver()
{
  if (_tao)
    TaoDestroy(&_tao);
}
//-----------------------------------------------------------------------------
void PETScTAOSolver::update_parameters(const std::string tao_type,
                                  const std::string ksp_type,
                                  const std::string pc_type)
{
  // Update parameters when tao/ksp/pc_types are explictly given
  if (tao_type != "default")
    parameters["method"] = tao_type;

  if (ksp_type != "default")
    parameters["linear_solver"] = ksp_type;

  if (pc_type != "default")
    parameters["preconditioner"] = pc_type;
}
//-----------------------------------------------------------------------------
void PETScTAOSolver::set_tao(const std::string tao_type)
{
  dolfin_assert(_tao);
  PetscErrorCode ierr;

  // Check that the requested method is known
  if (_methods.count(tao_type) == 0)
  {
    dolfin_error("PETScTAOSolver.cpp",
                 "set PETSc TAO solver",
                 "Unknown TAO method \"%s\"", tao_type.c_str());
  }

  // In case of an unconstrained minimisation problem, set the TAO
  // method to TAONTL
  if (!has_bounds && tao_type == "default")
  {
    ierr = TaoSetType(_tao, TAONTL);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetType");
  }
  else
  {
    // Set solver type
    std::map<std::string, std::pair<std::string,
                                    const TaoType>>::const_iterator it;
    it = _methods.find(tao_type);
    dolfin_assert(it != _methods.end());
    ierr = TaoSetType(_tao, it->second.second);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetType");
  }
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, bool>
PETScTAOSolver::solve(OptimisationProblem& optimisation_problem,
                      GenericVector& x,
                      const GenericVector& lb,
                      const GenericVector& ub)
{
  // Bound-constrained minimisation problem
  has_bounds = true;

  return solve(optimisation_problem, x.down_cast<PETScVector>(),
               lb.down_cast<PETScVector>(), ub.down_cast<PETScVector>());
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, bool>
PETScTAOSolver::solve(OptimisationProblem& optimisation_problem,
                      GenericVector& x)
{
  // Unconstrained minimisation problem
  has_bounds = false;
  PETScVector lb, ub;

  return solve(optimisation_problem, x.down_cast<PETScVector>(), lb, ub);
}
//-----------------------------------------------------------------------------
void PETScTAOSolver::init(OptimisationProblem& optimisation_problem,
                          PETScVector& x)
{
  // Unconstrained minimisation problem
  has_bounds = false;
  PETScVector lb, ub;
  init(optimisation_problem, x.down_cast<PETScVector>(), lb, ub);
}
//-----------------------------------------------------------------------------
void PETScTAOSolver::init(OptimisationProblem& optimisation_problem,
                          PETScVector& x,
                          const PETScVector& lb,
                          const PETScVector& ub)
{
  Timer timer("PETSc TAO solver init");
  PetscErrorCode ierr;

  // Form the optimisation problem object
  _tao_ctx.optimisation_problem = &optimisation_problem;

  // Set TAO/KSP parameters
  set_tao_options();
  set_ksp_options();

  // Initialise the Hessian matrix during the first call
  if (!_matH.mat())
  {
    PETScVector g;
    optimisation_problem.form(_matH, g, x);
    optimisation_problem.J(_matH, x);
  }
  dolfin_assert(_matH.mat());

  // Set initial vector
  ierr = TaoSetInitialVector(_tao, x.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetInitialVector");

  // Set the bounds in case of a bound-constrained minimisation problem
  if (has_bounds)
  {
    ierr = TaoSetVariableBounds(_tao, lb.vec(), ub.vec());
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetVariableBounds");
  }

  // Set the objective function, gradient and Hessian evaluation routines
  ierr = TaoSetObjectiveAndGradientRoutine(_tao, FormFunctionGradient, &_tao_ctx);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetObjectiveAndGradientRoutine");
  ierr = TaoSetHessianRoutine(_tao, _matH.mat(), _matH.mat(), FormHessian, &_tao_ctx);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetHessianRoutine");

  // Clear previous monitors
  ierr = TaoCancelMonitors(_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoCancelMonitors");

  // Set the monitor
  if (parameters["monitor_convergence"])
  {
    ierr = TaoSetMonitor(_tao, TaoDefaultMonitor,
                         PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)_tao)),
                         NULL);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetMonitor");
  }

  // Check for any TAO command line options
  std::string prefix = std::string(parameters["options_prefix"]);
  if (prefix != "default")
  {
    // Make sure that the prefix has a '_' at the end if the user
    // didn't provide it
    char lastchar = *prefix.rbegin();
    if (lastchar != '_')
      prefix += "_";
    ierr = TaoSetOptionsPrefix(_tao, prefix.c_str());
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetOptionsPrefix");
  }
  ierr = TaoSetFromOptions(_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetFromOptions");

  // Set the convergence test
  ierr = TaoSetConvergenceTest(_tao, TaoConvergenceTest, this);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetConvergenceTest");
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, bool>
PETScTAOSolver::solve(OptimisationProblem& optimisation_problem,
                      PETScVector& x,
                      const PETScVector& lb,
                      const PETScVector& ub)
{
  Timer timer("PETSc TAO solver execution");
  PetscErrorCode ierr;

  // Initialise the TAO solver
  PETScVector x_copy(x);
  init(optimisation_problem, x_copy, lb, ub);

  // Solve
  ierr = TaoSolve(_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSolve");

  // Get the solution vector
  x.zero();
  x.axpy(1.0, x_copy);

  // Update ghost values
  x.update_ghost_values();

  // Print the report on convergence and methods used
  if (parameters["report"])
  {
    ierr = TaoView(_tao, PETSC_VIEWER_STDOUT_WORLD);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoView");
  }

  // Check for convergence
  TaoConvergedReason reason;
  ierr = TaoGetConvergedReason(_tao, &reason);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetConvergedReason");

  // Get the number of iterations
  PetscInt its = 0;
  ierr = TaoGetIterationNumber(_tao, &its);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetIterationNumber");

  // Report number of iterations
  if (reason >= 0)
    log(PROGRESS, "TAO solver converged\n");
  else
  {
    bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
    if (error_on_nonconvergence)
    {
      ierr = TaoView(_tao, PETSC_VIEWER_STDOUT_WORLD);
      if (ierr != 0) petsc_error(ierr, __FILE__, "TaoView");
      dolfin_error("PETScTAOSolver.cpp",
                   "solve nonlinear optimisation problem",
                   "Solution failed to converge in %i iterations (TAO reason %d)",
                   its, reason);
    }
    else
    {
      log(WARNING, "TAO solver failed to converge. Try a different TAO method" \
                   " or adjust some parameters.");
    }
  }

  return std::make_pair(its, reason > 0);
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScTAOSolver::FormFunctionGradient(Tao tao, Vec x,
                                                    PetscReal *fobj, Vec g,
                                                    void *ctx)
{
  // Get the optimisation problem object
  struct tao_ctx_t tao_ctx = *(struct tao_ctx_t*) ctx;
  OptimisationProblem* optimisation_problem = tao_ctx.optimisation_problem;

  // Wrap the PETSc objects
  PETScVector x_wrap(x);
  PETScVector g_wrap(g);

  // Compute the objective function f and its gradient g = f'
  PETScMatrix H;
  *fobj = optimisation_problem->f(x_wrap);
  optimisation_problem->form(H, g_wrap, x_wrap);
  optimisation_problem->F(g_wrap, x_wrap);

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
  PETScMatrix H_wrap(H);

  // Compute the hessian H(x) = f''(x)
  PETScVector g;
  optimisation_problem->form(H_wrap, g, x_wrap);
  optimisation_problem->J(H_wrap, x_wrap);

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScTAOSolver::TaoConvergenceTest(Tao tao, void *ctx)
{
  PetscInt its;
  PetscReal f, gnorm, cnorm, xdiff;
  TaoConvergedReason reason;
  PetscErrorCode ierr;
  ierr = TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetSolutionStatus");

  // We enforce Tao to do at least one iteration
  if (its < 1)
  {
    ierr = TaoSetConvergedReason(tao, TAO_CONTINUE_ITERATING);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetConvergedReason");
  }
  else
  {
    ierr = TaoDefaultConvergenceTest(tao, &ctx);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoDefaultConvergenceTest");
  }

  return 0;
}
//------------------------------------------------------------------------------
void PETScTAOSolver::set_tao_options()
{
  dolfin_assert(_tao);
  PetscErrorCode ierr;

  // Set the TAO solver
  set_tao(parameters["method"]);

  // Set tolerances
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 6 && PETSC_VERSION_RELEASE == 1
  ierr = TaoSetTolerances(_tao, parameters["function_absolute_tol"],
                                parameters["function_relative_tol"],
                                parameters["gradient_absolute_tol"],
                                parameters["gradient_relative_tol"],
                                parameters["gradient_t_tol"]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetTolerances");
  #else
  ierr = TaoSetTolerances(_tao, parameters["gradient_absolute_tol"],
                                parameters["gradient_relative_tol"],
                                parameters["gradient_t_tol"]);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetTolerances");
  #endif

  // Set TAO solver maximum iterations
  int maxits = parameters["maximum_iterations"];
  ierr = TaoSetMaximumIterations(_tao, maxits);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetMaximumIterations");

  // Set TAO line search
  const std::string line_search_type = parameters["line_search"];
  if (line_search_type != "default")
  {
    TaoLineSearch linesearch;
    ierr = TaoGetLineSearch(_tao, &linesearch);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetLineSearch");
    ierr = TaoLineSearchSetType(linesearch, line_search_type.c_str());
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoLineSearchSetType");
  }
}
//-----------------------------------------------------------------------------
void PETScTAOSolver::set_ksp_options()
{
  dolfin_assert(_tao);
  PetscErrorCode ierr;
  KSP ksp;
  ierr = TaoGetKSP(_tao, &ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetKSP");
  const std::string ksp_type  = parameters["linear_solver"];
  const std::string pc_type = parameters["preconditioner"];

  // Set the KSP solver and its options
  if (ksp)
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

    if (ksp_type == "default")
    {
      // Do nothing
    }

    // Set type for iterative Krylov solver
    else if (PETScKrylovSolver::_methods.count(ksp_type) != 0)
    {
      std::map<std::string, const KSPType>::const_iterator ksp_pair
        = PETScKrylovSolver::_methods.find(ksp_type);
      dolfin_assert(ksp_pair != PETScKrylovSolver::_methods.end());
      ierr = KSPSetType(ksp, ksp_pair->second);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");

      if (pc_type != "default")
      {
        std::map<std::string, const PCType>::const_iterator pc_pair
          = PETScPreconditioner::_methods.find(pc_type);
        dolfin_assert(pc_pair != PETScPreconditioner::_methods.end());
        ierr = PCSetType(pc, pc_pair->second);
        if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");
      }
    }
    else if (ksp_type == "lu" || PETScLUSolver::_methods.count(ksp_type) != 0)
    {
      std::string lu_method;
      if (PETScLUSolver::_methods.find(ksp_type) != PETScLUSolver::_methods.end())
      {
        lu_method = ksp_type;
      }
      else
      {
        MPI_Comm comm = MPI_COMM_NULL;
        PetscObjectGetComm((PetscObject)_tao, &comm);
        if (MPI::size(comm) == 1)
        {
          #if PETSC_HAVE_UMFPACK
          lu_method = "umfpack";
          #elif PETSC_HAVE_MUMPS
          lu_method = "mumps";
          #elif PETSC_HAVE_PASTIX
          lu_method = "pastix";
          #elif PETSC_HAVE_SUPERLU
          lu_method = "superlu";
          #else
          lu_method = "petsc";
          warning("Using PETSc native LU solver. Consider configuring PETSc with an efficient LU solver (e.g. UMFPACK, MUMPS).");
          #endif
        }
        else
        {
          #if PETSC_HAVE_SUPERLU_DIST
          lu_method = "superlu_dist";
          #elif PETSC_HAVE_PASTIX
          lu_method = "pastix";
          #elif PETSC_HAVE_MUMPS
          lu_method = "mumps";
          #else
          dolfin_error("PETScTAOSolver.cpp",
                       "solve linear system using PETSc LU solver",
                       "No suitable solver for parallel LU found. Consider configuring PETSc with MUMPS or SuperLU_dist");
          #endif
        }
      }
      ierr = KSPSetType(ksp, KSPPREONLY);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");
      ierr = PCSetType(pc, PCLU);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCSetType");
      std::map<std::string, const MatSolverPackage>::const_iterator lu_pair
        = PETScLUSolver::_methods.find(lu_method);
      dolfin_assert(lu_pair != PETScLUSolver::_methods.end());
      ierr = PCFactorSetMatSolverPackage(pc, lu_pair->second);
      if (ierr != 0) petsc_error(ierr, __FILE__, "PCFactorSetMatSolverPackage");
    }
    else     // Unknown KSP method
    {
      dolfin_error("PETScTAOSolver.cpp",
                   "set linear solver options",
                   "Unknown KSP method \"%s\"", ksp_type.c_str());
    }

    // In any case, set the KSP options specified by the user
    Parameters krylov_parameters = parameters("krylov_solver");

    // Non-zero initial guess
    const bool nonzero_guess = krylov_parameters["nonzero_initial_guess"];
    if (nonzero_guess)
    {
      ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetInitialGuessNonzero");
    }
    else
    {
      ierr = KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetInitialGuessNonzero");
    }

    // KSP monitor
    if (krylov_parameters["monitor_convergence"])
    {
      ierr = KSPMonitorSet(ksp, KSPMonitorTrueResidualNorm,
                           PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp)),
                           NULL);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPMonitorSet");
    }

    // Get integer tolerances (to take care of casting to PetscInt)
    const int max_iter = krylov_parameters["maximum_iterations"];

    // Set tolerances
    ierr = KSPSetTolerances(ksp,
                            krylov_parameters["relative_tolerance"],
                            krylov_parameters["absolute_tolerance"],
                            krylov_parameters["divergence_limit"],
                            max_iter);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetTolerances");
  }
  else
  {
    warning("The underlying linear solver cannot be modified for this specified TAO solver. The options are all ignored.");
  }
}
//-----------------------------------------------------------------------------

#endif
