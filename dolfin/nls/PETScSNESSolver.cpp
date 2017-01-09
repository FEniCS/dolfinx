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
// Modified by Corrado Maurini 2013
// Modified by Anders Logg 2013
//
// First added:  2012-10-13
// Last changed: 2013-11-21

#ifdef HAS_PETSC

#include <map>
#include <string>
#include <utility>
#include <cmath>
#include <petscsys.h>

#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScLUSolver.h>
#include <dolfin/la/PETScPreconditioner.h>
#include <dolfin/la/PETScVector.h>
#include "NonlinearProblem.h"
#include "PETScSNESSolver.h"

using namespace dolfin;

// Mapping from method string to PETSc
const std::map<std::string, std::pair<std::string, const SNESType>>
PETScSNESSolver::_methods
= { {"default",      {"default SNES method", ""}},
    {"newtonls",     {"Line search method", SNESNEWTONLS}},
    {"newtontr",     {"Trust region method", SNESNEWTONTR}},
    {"test",         {"Tool to verify Jacobian approximation", SNESTEST}},
    {"ngmres",       {"Nonlinear generalised minimum residual method",
                      SNESNGMRES}},
    {"nrichardson",  {"Richardson nonlinear method (Picard iteration)",
                      SNESNRICHARDSON}},
    {"vinewtonrsls", {"Reduced space active set solver method (for bounds)",
                      SNESVINEWTONRSLS}},
    {"vinewtonssls", {"Reduced space active set solver method (for bounds)",
                      SNESVINEWTONSSLS}},
    {"qn",           {"Limited memory quasi-Newton", SNESQN}},
    {"ncg",          {"Nonlinear conjugate gradient method", SNESNCG}},
    {"fas",          {"Full Approximation Scheme nonlinear multigrid method",
                      SNESFAS}},
    {"nasm",         {"Nonlinear Additive Schwartz", SNESNASM}},
    {"anderson",     {"Anderson mixing method", SNESANDERSON}},
    {"aspin",        {"Additive-Schwarz Preconditioned Inexact Newton",
                      SNESASPIN}},
    {"ms",           {"Multistage smoothers", SNESMS}} };

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string>> PETScSNESSolver::methods()
{
  std::vector<std::pair<std::string, std::string>> available_methods;
  for (auto it = _methods.begin(); it != _methods.end(); ++it)
    available_methods.push_back(std::make_pair(it->first, it->second.first));
  return available_methods;
}
//-----------------------------------------------------------------------------
Parameters PETScSNESSolver::default_parameters()
{
  Parameters p(NewtonSolver::default_parameters());
  p.rename("snes_solver");
  p.add("solution_tolerance", 1.0e-16);
  p.add("maximum_residual_evaluations", 2000);
  p.remove("convergence_criterion");
  p.remove("relaxation_parameter");
  p.remove("method");
  p.add("method", "default");
  p.add("line_search", "basic",  {"basic", "bt", "l2", "cp", "nleqerr"});
  p.add("sign", "default", {"default", "nonnegative", "nonpositive"});

  return p;
}
//-----------------------------------------------------------------------------
PETScSNESSolver::PETScSNESSolver(MPI_Comm comm) : _snes(nullptr),
                                                  _has_explicit_bounds(false)
{
  // Create SNES object
  SNESCreate(comm, &_snes);

  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
PETScSNESSolver::PETScSNESSolver(std::string nls_type) : _snes(nullptr),
                                                         _has_explicit_bounds(false)
{
  // Create SNES object
  SNESCreate(MPI_COMM_WORLD, &_snes);

  const std::map<std::string, std::pair<std::string, const SNESType>>::const_iterator
    method = _methods.find(nls_type);

  // Check that the requested method is known
  if (method != _methods.end())
  {
    // Set solver type
    if (method->first != "default")
      SNESSetType(_snes, method->second.second);
  }
  else
  {
    dolfin_error("PETScSNESSolver.cpp",
                 "create PETSc SNES solver",
                 "Unknown SNES method \"%s\"", nls_type.c_str());
  }

  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
PETScSNESSolver::~PETScSNESSolver()
{
  if (_snes)
    SNESDestroy(&_snes);
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, bool>
PETScSNESSolver::solve(NonlinearProblem& nonlinear_problem,
                       GenericVector& x,
                       const GenericVector& lb,
                       const GenericVector& ub)
{
  // Set linear solver parameters
  set_linear_solver_parameters();

  // Check size of the bound vectors
  if (lb.size() != ub.size())
  {
    dolfin_error("PETScSNESSolver.cpp",
                 "assigning upper and lower bounds",
                 "The size of the given upper and lower bounds is different");
  }
  else if (lb.size() != x.size())
  {
    dolfin_error("PETScSNESSolver.cpp",
                 "assigning upper and lower bounds",
                 "The size of the bounds is different from the size of the solution vector");
  }

  // Set the bounds
  std::shared_ptr<const PETScVector>
    _ub(&ub.down_cast<PETScVector>(), NoDeleter());
  std::shared_ptr<const PETScVector>
    _lb(&lb.down_cast<PETScVector>(), NoDeleter());
  this->lb = _lb;
  this->ub = _ub;
  _has_explicit_bounds = true;

  return this->solve(nonlinear_problem, x);
}
//-----------------------------------------------------------------------------
void PETScSNESSolver::init(NonlinearProblem& nonlinear_problem,
                           GenericVector& x)
{
  Timer timer("SNES solver init");
  PETScMatrix A(this->mpi_comm());

  // Set linear solver parameters
  set_linear_solver_parameters();

  _snes_ctx.nonlinear_problem = &nonlinear_problem;
  _snes_ctx.x = &x.down_cast<PETScVector>();
  VecDuplicate(_snes_ctx.x->vec(), &_snes_ctx.f_tmp);

  // Compute F(u)
  PETScVector f(_snes_ctx.f_tmp);
  nonlinear_problem.form(A, f, x);
  nonlinear_problem.F(f, x);
  nonlinear_problem.J(A, x);

  SNESSetFunction(_snes, _snes_ctx.f_tmp, PETScSNESSolver::FormFunction,
                  &_snes_ctx);
  SNESSetJacobian(_snes, A.mat(), A.mat(), PETScSNESSolver::FormJacobian,
                  &_snes_ctx);
  SNESSetObjective(_snes, PETScSNESSolver::FormObjective, &_snes_ctx);

  // Set some options from the parameters
  if (parameters["report"].is_set())
  {
    if (parameters["report"])
    {
      PetscErrorCode ierr;
      PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)_snes));
      PetscViewerFormat format = PETSC_VIEWER_DEFAULT;
      PetscViewerAndFormat *vf;
      PetscViewerAndFormatCreate(viewer,format,&vf);
      ierr = SNESMonitorSet(_snes,
                            (PetscErrorCode (*)(SNES,PetscInt,PetscReal,void*)) SNESMonitorDefault,
                            vf,
                            (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
      if (ierr != 0) petsc_error(ierr, __FILE__, "SNESMonitorSet");
    }
  }

  // Set the bounds, if any
  set_bounds(x);

  // Set the method
  const std::string method = parameters["method"];
  if (method != "default")
  {
    auto it = _methods.find(method);
    dolfin_assert(it != _methods.end());
    SNESSetType(_snes, it->second.second);

    // Check if bounds/sign are set when VI method requested
    if ((method == "vinewtonrsls" || method == "vinewtonssls") && !is_vi())
    {
      dolfin_error("PETScSNESSolver.cpp",
                   "set up SNES VI solver",
                   "Need to set bounds or sign for vinewtonrsls or vinewtonssls"
                   " methods");
    }
  }
  else if (method == "default" && is_vi())
  {
    // If
    //      a) the user has set bounds (is_vi())
    // AND  b) the user has not set a solver (method == default)
    // THEN set a good method that supports bounds
    // (most methods do not support bounds)
    auto it = _methods.find("vinewtonssls");
    dolfin_assert(it != _methods.end());
    SNESSetType(_snes, it->second.second);
  }

  SNESLineSearch linesearch;
  SNESGetLineSearch(_snes, &linesearch);

  if (parameters["report"].is_set())
  {
    if (parameters["report"])
      SNESLineSearchMonitor(linesearch);
  }

  const std::string line_search_type = std::string(parameters["line_search"]);
  SNESLineSearchSetType(linesearch, line_search_type.c_str());

  // Tolerances
  const int max_iters = parameters["maximum_iterations"];
  const int max_residual_evals = parameters["maximum_residual_evaluations"];
  SNESSetTolerances(_snes, parameters["absolute_tolerance"],
                    parameters["relative_tolerance"],
                    parameters["solution_tolerance"],
                    max_iters, max_residual_evals);

  // Set some options
  SNESSetFromOptions(_snes);
}
//-----------------------------------------------------------------------------
void PETScSNESSolver::set_options_prefix(std::string options_prefix)
{
  // Set options prefix
  dolfin_assert(_snes);
  PetscErrorCode ierr = SNESSetOptionsPrefix(_snes, options_prefix.c_str());
  if (ierr != 0) petsc_error(ierr, __FILE__, "SNESSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string PETScSNESSolver::get_options_prefix() const
{
  dolfin_assert(_snes);
  const char* prefix = NULL;
  PetscErrorCode ierr = SNESGetOptionsPrefix(_snes, &prefix);
  if (ierr != 0) petsc_error(ierr, __FILE__, "SNESGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void PETScSNESSolver::set_from_options() const
{
  dolfin_assert(_snes);
  PetscErrorCode ierr = SNESSetFromOptions(_snes);
  if (ierr != 0) petsc_error(ierr, __FILE__, "SNESSetFromOptions");
}
//-----------------------------------------------------------------------------
MPI_Comm PETScSNESSolver::mpi_comm() const
{
  dolfin_assert(_snes);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscObjectGetComm((PetscObject)_snes, &mpi_comm);
  return mpi_comm;
}
//-----------------------------------------------------------------------------
std::pair<std::size_t, bool>
PETScSNESSolver::solve(NonlinearProblem& nonlinear_problem,
                       GenericVector& x)
{
  Timer timer("SNES solver execution");
  PETScVector f;
  PETScMatrix A;
  PetscInt its;
  SNESConvergedReason reason;

  this->init(nonlinear_problem, x);
  // for line searches, making this copy is necessary:
  // when linesearching, the working space can't be the
  // same as the vector that holds the current solution
  // guess in the dolfin form.
  PETScVector x_copy(x.down_cast<PETScVector>());
  SNESSolve(_snes, NULL, x_copy.vec());
  x.zero();
  x.axpy(1.0, x_copy);

  // Update any ghost values
  PETScVector& _x = x.down_cast<PETScVector>();
  _x.update_ghost_values();

  SNESGetIterationNumber(_snes, &its);
  SNESGetConvergedReason(_snes, &reason);

  const bool report = parameters["report"];

  MPI_Comm comm = mpi_comm();
  if (reason > 0 && report && dolfin::MPI::rank(comm) == 0)
  {
    info("PETSc SNES solver converged in %d iterations with convergence reason %s.",
         its, SNESConvergedReasons[reason]);
  }
  else if (reason < 0 && dolfin::MPI::rank(comm) == 0)
  {
    warning("PETSc SNES solver diverged in %d iterations with divergence reason %s.",
            its, SNESConvergedReasons[reason]);
  }

  if (parameters["error_on_nonconvergence"] && reason < 0)
  {
    dolfin_error("PETScSNESSolver.cpp",
                 "solve nonlinear system with PETScSNESSolver",
                 "Solver did not converge",
                 "Bummer");
  }

  return std::make_pair(its, reason > 0);
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScSNESSolver::FormFunction(SNES snes, Vec x, Vec f, void* ctx)
{
  auto snes_ctx = static_cast<struct snes_ctx_t*>(ctx);
  NonlinearProblem* nonlinear_problem = snes_ctx->nonlinear_problem;
  PETScVector* _x = snes_ctx->x;

  // Wrap the PETSc Vec as DOLFIN PETScVector
  PETScVector x_wrap(x);
  PETScVector f_wrap(f);

  // Update current solution that is associated with nonlinear
  // problem. This is required because x is not the solution vector
  // that was passed to PETSc. PETSc updates the solution vector at
  // the end of solve. We should find a better solution.
  *_x = x_wrap;

  // Update ghost values
  _x ->update_ghost_values();

  // Compute F(u)
  PETScMatrix A(_x->mpi_comm());
  nonlinear_problem->form(A, f_wrap, *_x);
  nonlinear_problem->F(f_wrap, *_x);

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScSNESSolver::FormObjective(SNES snes, Vec x,
                                              PetscReal* out, void* ctx)
{
  auto snes_ctx = static_cast<struct snes_ctx_t*>(ctx);
  PETScSNESSolver::FormFunction(snes, x, snes_ctx->f_tmp, ctx);
  PetscReal f_norm;
  VecNorm(snes_ctx->f_tmp, NORM_2, &f_norm);

  if (std::isnan(f_norm) || std::isinf(f_norm))
    *out = 1.0e100;
  else
    *out = f_norm;

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode PETScSNESSolver::FormJacobian(SNES snes, Vec x, Mat A, Mat P,
                                             void* ctx)
{
  // Interface does not presently support a preconditioner that
  // differs from operator A
  if (A != P)
  {
    dolfin_error("PETScSNESSolver.cpp",
                 "for Jacobian",
                 "Matrix object incompatibility. The Jacobian matrix must not be reset when using PETSc SNES.");
  }

  // Get nonlinear problem object
  auto snes_ctx = static_cast<struct snes_ctx_t*>(ctx);
  NonlinearProblem* nonlinear_problem = snes_ctx->nonlinear_problem;

  // Wrap the PETSc objects
  PETScMatrix A_wrap(P);
  PETScVector x_wrap(x);

  // Form Jacobian
  PETScVector f(x_wrap.mpi_comm());
  nonlinear_problem->form(A_wrap, f, x_wrap);
  nonlinear_problem->J(A_wrap, x_wrap);

  return 0;
}
//-----------------------------------------------------------------------------
void PETScSNESSolver::set_linear_solver_parameters()
{
  KSP ksp;
  PC pc;

  PetscErrorCode ierr;
  ierr = SNESGetKSP(_snes, &ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "SNESGetKSP");

  ierr = KSPGetPC(ksp, &pc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");

  // Get MPI communicator
  MPI_Comm comm = mpi_comm();

  if (parameters["report"].is_set())
  {
    if (parameters["report"])
    {
      PetscViewer viewer = PETSC_VIEWER_STDOUT_(comm);
      PetscViewerFormat format = PETSC_VIEWER_DEFAULT;
      PetscViewerAndFormat *vf;
      ierr = PetscViewerAndFormatCreate(viewer,format,&vf);
      ierr = KSPMonitorSet(ksp,
                           (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*)) KSPMonitorDefault,
                           vf,
                           (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
    }
  }
  const std::string linear_solver  = parameters["linear_solver"];
  const std::string preconditioner = parameters["preconditioner"];

  if (linear_solver == "default")
  {
    // Do nothing
  }
  else if (PETScKrylovSolver::_methods.count(linear_solver) != 0)
  {
    auto  solver_pair = PETScKrylovSolver::_methods.find(linear_solver);
    dolfin_assert(solver_pair != PETScKrylovSolver::_methods.end());
    KSPSetType(ksp, solver_pair->second);
    if (preconditioner != "default"
        && PETScPreconditioner::_methods.count(preconditioner) != 0)
    {
      auto it = PETScPreconditioner::_methods.find(preconditioner);
      dolfin_assert(it != PETScPreconditioner::_methods.end());
      PCSetType(pc, it->second);
    }

    Parameters krylov_parameters = parameters("krylov_solver");

    // Set non-zero initial guess...
    if (krylov_parameters["nonzero_initial_guess"].is_set())
    {
      const bool nonzero_guess = krylov_parameters["nonzero_initial_guess"];
      if (nonzero_guess)
        KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
      else
        KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    }
    // ... otherwise use the default, what has been set before,
    // or directly through PETSc API or parameter system

    if (krylov_parameters["monitor_convergence"].is_set()
        && krylov_parameters["monitor_convergence"])
    {
      PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
      PetscViewerFormat format = PETSC_VIEWER_DEFAULT;
      PetscViewerAndFormat *vf;
      ierr = PetscViewerAndFormatCreate(viewer,format,&vf);
      ierr = KSPMonitorSet(ksp,
                         (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*)) KSPMonitorTrueResidualNorm,
                         vf,
                         (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
    }

    // Check if a tolerance has been set
    if (krylov_parameters["relative_tolerance"].is_set()
        or krylov_parameters["absolute_tolerance"].is_set()
        or krylov_parameters["divergence_limit"].is_set()
        or krylov_parameters["maximum_iterations"].is_set())
    {
      // Set tolerances
      const double rtol = krylov_parameters["relative_tolerance"].is_set() ? (double)krylov_parameters["relative_tolerance"] : PETSC_DEFAULT;
      const double atol = krylov_parameters["absolute_tolerance"].is_set() ? (double)krylov_parameters["absolute_tolerance"] : PETSC_DEFAULT;
      const double dtol = krylov_parameters["divergence_limit"].is_set() ? (double)krylov_parameters["divergence_limit"] : PETSC_DEFAULT;
      const int max_it  = krylov_parameters["maximum_iterations"].is_set() ? (int)krylov_parameters["maximum_iterations"] : PETSC_DEFAULT;
      ierr = KSPSetTolerances(ksp, rtol, atol, dtol, max_it);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetTolerances");
    }
  }
  else if (linear_solver == "lu"
           || PETScLUSolver::_methods.count(linear_solver) != 0)
  {
    std::string lu_method;
    if (PETScLUSolver::_methods.find(linear_solver)
        != PETScLUSolver::_methods.end())
    {
      lu_method = linear_solver;
    }
    else
    {
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
        dolfin_error("PETScSNESSolver.cpp",
                     "solve linear system using PETSc LU solver",
                     "No suitable solver for parallel LU found. Consider configuring PETSc with MUMPS or SuperLU_dist");
        #endif
      }
    }

    KSPSetType(ksp, KSPPREONLY);
    PCSetType(pc, PCLU);
    auto it = PETScLUSolver::_methods.find(lu_method);
    dolfin_assert(it != PETScLUSolver::_methods.end());
    PCFactorSetMatSolverPackage(pc, it->second);
  }
  else
  {
    dolfin_error("PETScSNESSolver.cpp",
                 "set linear solver options",
                 "Unknown KSP method \"%s\"", linear_solver.c_str());
  }
}
//-----------------------------------------------------------------------------
void PETScSNESSolver::set_bounds(GenericVector& x)
{
  if (is_vi())
  {
    dolfin_assert(_snes);
    const std::string sign   = parameters["sign"];
    const std::string method = parameters["method"];
    if (method != "vinewtonrsls" && method != "vinewtonssls" && method != "default")
    {
      dolfin_error("PETScSNESSolver.cpp",
                   "set variational inequality bounds",
                   "Need to use vinewtonrsls or vinewtonssls methods if bounds are set");
    }

    if (sign != "default")
    {
      // Here, x is the model vector from which we make our Vecs that
      // tell PETSc the bounds.
      Vec ub, lb;

      PETScVector _x = x.down_cast<PETScVector>();
      VecDuplicate(_x.vec(), &ub);
      VecDuplicate(_x.vec(), &lb);
      if (sign == "nonnegative")
      {
        VecSet(ub, PETSC_INFINITY);
        VecSet(lb, 0.0);
      }
      else if (sign == "nonpositive")
      {
        VecSet(ub, 0.0);
        VecSet(lb, PETSC_INFINITY);
      }
      else
      {
        dolfin_error("PETScSNESSolver.cpp",
                     "set PETSc SNES solver bounds",
                     "Unknown bound type \"%s\"", sign.c_str());
      }

      SNESVISetVariableBounds(_snes, lb, ub);
      VecDestroy(&ub);
      VecDestroy(&lb);
    }
    else if (_has_explicit_bounds)
    {
      const PETScVector* lb = this->lb.get();
      const PETScVector* ub = this->ub.get();
      SNESVISetVariableBounds(_snes, lb->vec(), ub->vec());
    }
  }
}
//-----------------------------------------------------------------------------
bool PETScSNESSolver::is_vi() const
{
  const std::string sign = parameters["sign"];
  if (sign != "default" && this->_has_explicit_bounds)
  {
    dolfin_error("PETScSNESSolver.cpp",
                 "set variational inequality bounds",
                 "Both the sign parameter and the explicit bounds are set");
    return false;
  }
  else if (sign != "default" || this->_has_explicit_bounds)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------

#endif
