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

#ifdef HAS_PETSC

#include <petsclog.h>
#include <petscversion.h>

#include <dolfin/common/Timer.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/log.h>
#include "dolfin/la/GenericMatrix.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/la/PETScMatrix.h"
#include "dolfin/la/PETScVector.h"
#include "dolfin/la/PETScKrylovSolver.h"
#include "dolfin/la/PETScPreconditioner.h"
#include "TAOLinearBoundSolver.h"
#include "petscksp.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petsctao.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// Mapping from ksp_method string to PETSc
const std::map<std::string, const KSPType> TAOLinearBoundSolver::_ksp_methods
= { {"default",    ""},
    {"cg",         KSPCG},
    {"gmres",      KSPGMRES},
    {"minres",     KSPMINRES},
    {"tfqmr",      KSPTFQMR},
    {"richardson", KSPRICHARDSON},
    {"nash",       KSPNASH},
    {"stcg",       KSPSTCG},
    {"bicgstab",   KSPBCGS} };
//-----------------------------------------------------------------------------
// Mapping from method string to description
const std::map<std::string, std::string> TAOLinearBoundSolver::_methods_descr
= { {"default"  ,  "Default Tao method (tao_tron)"},
    {"tron" ,  "Newton Trust Region method"},
    {"bqpip",  "Interior Point Newton Algorithm"},
    {"gpcg" ,  "Gradient Projection Conjugate Gradient"},
    {"blmvm",  "Limited memory variable metric method"} };
//-----------------------------------------------------------------------------
std::map<std::string, std::string> TAOLinearBoundSolver::methods()
{
  return TAOLinearBoundSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> TAOLinearBoundSolver::krylov_solvers()
{
  return PETScKrylovSolver::methods();
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> TAOLinearBoundSolver::preconditioners()
{
  return PETScPreconditioner::preconditioners();
}
//-----------------------------------------------------------------------------
TAOLinearBoundSolver::TAOLinearBoundSolver(MPI_Comm comm)
  : _tao(nullptr), _preconditioner_set(false)
{
  PetscErrorCode ierr;

  // Create TAO object
  ierr = TaoCreate(PETSC_COMM_WORLD, &_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoCreate");
}
//-----------------------------------------------------------------------------
TAOLinearBoundSolver::TAOLinearBoundSolver(const std::string method,
                                           const std::string ksp_type,
                                           const std::string pc_type)
  : _tao(NULL), _preconditioner(new PETScPreconditioner(pc_type)),
    _preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  PetscErrorCode ierr;

  // Create TAO object
  ierr = TaoCreate(PETSC_COMM_WORLD, &_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoCreate");

  // Set tao solver
  set_solver(method);

  //Set the PETSC KSP used by TAO
  set_ksp(ksp_type);

  // Some preconditioners may lead to errors because not compatible with TAO.
  if ((pc_type != "default") or (ksp_type != "default")
      or (method != "default"))
  {
    log(WARNING, "Some preconditioners may be not be applicable to "\
    "TAO solvers and generate errors.");
  }
}
//-----------------------------------------------------------------------------
TAOLinearBoundSolver::~TAOLinearBoundSolver()
{
  if (_tao)
    TaoDestroy(&_tao);
}
//-----------------------------------------------------------------------------
void
TAOLinearBoundSolver::set_operators(std::shared_ptr<const GenericMatrix> A,
                                    std::shared_ptr<const GenericVector> b)
{
  std::shared_ptr<const PETScMatrix>
    _matA = GenericTensor::down_cast<const PETScMatrix>(A);
  std::shared_ptr<const PETScVector>
    _b = GenericTensor::down_cast<const PETScVector>(b);
  set_operators(_matA, _b);
}
//-----------------------------------------------------------------------------
void
TAOLinearBoundSolver::set_operators(std::shared_ptr<const PETScMatrix> A,
                                    std::shared_ptr<const PETScVector> b)
{
  this->_matA = A;
  this->_b = b;
}
//-----------------------------------------------------------------------------
std::size_t TAOLinearBoundSolver::solve(const GenericMatrix& A1,
                                        GenericVector& x,
                                        const GenericVector& b1,
                                        const GenericVector& xl,
                                        const GenericVector& xu)
{
  return solve(A1.down_cast<PETScMatrix>(),
               x.down_cast<PETScVector>(),
               b1.down_cast<PETScVector>(),
               xl.down_cast<PETScVector>(),
               xu.down_cast<PETScVector>());
}
//-----------------------------------------------------------------------------
std::size_t TAOLinearBoundSolver::solve(const PETScMatrix& A1,
                                        PETScVector& x,
                                        const PETScVector& b1,
                                        const PETScVector& xl,
                                        const PETScVector& xu)
{
  PetscErrorCode ierr;

  // Check symmetry
  dolfin_assert(A1.size(0) == A1.size(1));

  // Set operators (A and b)
  std::shared_ptr<const PETScMatrix> A(&A1, NoDeleter());
  std::shared_ptr<const PETScVector> b(&b1, NoDeleter());
  set_operators(A, b);
  dolfin_assert(A->mat());
  //dolfin_assert(b->vec());

  // Set initial vector
  dolfin_assert(_tao);
  ierr = TaoSetInitialVector(_tao, x.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetInitialVector");

  // Set the bound on the variables
  ierr = TaoSetVariableBounds(_tao, xl.vec(), xu.vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetVariableBounds");

  // Set the user function, gradient, hessian evaluation routines and
  // data structures
  ierr = TaoSetObjectiveAndGradientRoutine(_tao,
           __TAOFormFunctionGradientQuadraticProblem, this);
  if (ierr != 0) petsc_error(ierr, __FILE__,
                             "TaoSetObjectiveAndGradientRoutine");
  ierr = TaoSetHessianRoutine(_tao, A->mat(), A->mat(),
                              __TAOFormHessianQuadraticProblem, this);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetHessianRoutine");

  // Set parameters from local parameters, including ksp parameters
  read_parameters();

  // Clear previous monitors
  ierr = TaoCancelMonitors(_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoCancelMonitors");

  // Set the monitor
  if (parameters["monitor_convergence"].is_set())
  {
    if (parameters["monitor_convergence"])
    {
      ierr = TaoSetMonitor(_tao, __TAOMonitor, this, NULL);
      if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetMonitor");
    }
  }

  // Solve the bound constrained problem
  Timer timer("TAO solver");
  const char* tao_type;
  ierr = TaoGetType(_tao, &tao_type);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetType");
  log(PROGRESS, "Tao solver %s starting to solve %i x %i system", tao_type,
      A->size(0), A->size(1));

  // Solve
  ierr = TaoSolve(_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSolve");

  // Update ghost values
  x.update_ghost_values();

  // Print the report on convergences and methods used
  if (parameters["report"].is_set())
  {
    if (parameters["report"])
    {
      ierr = TaoView(_tao, PETSC_VIEWER_STDOUT_WORLD);
      if (ierr != 0) petsc_error(ierr, __FILE__, "TaoView");
    }
  }

  // Check for convergence
  TaoConvergedReason reason;
  ierr = TaoGetConvergedReason(_tao, &reason);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetConvergedReason");

  // Get the number of iterations
  PetscInt num_iterations = 0;
  ierr = TaoGetMaximumIterations(_tao, &num_iterations);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetMaximumIterations");

  // Report number of iterations
  if (reason >= 0)
    log(PROGRESS, "Tao solver converged\n");
  else
  {
    if (parameters["error_on_nonconvergence"].is_set())
    {
      bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
      if (error_on_nonconvergence)
      {
        ierr = TaoView(_tao, PETSC_VIEWER_STDOUT_WORLD);
        if (ierr != 0) petsc_error(ierr, __FILE__, "TaoView");
        dolfin_error("TAOLinearBoundSolver.cpp",
                     "solve linear system using Tao solver",
                     "Solution failed to converge in %i iterations (TAO reason %d)",
                     num_iterations, reason);
      }
      else
      {
        log(WARNING,  "Tao solver %s failed to converge. Try a different TAO method," \
            " adjust some parameters", tao_type);
      }
    }
  }

  return num_iterations;
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_solver(const std::string& method)
{
  dolfin_assert(_tao);
  PetscErrorCode ierr;

  // Do nothing if default type is specified
  if (method == "default")
    ierr = TaoSetType(_tao, "tron");
  else
  {
    // Choose solver
    if (method == "tron")
      ierr = TaoSetType(_tao, "tron");
    else if (method == "blmvm")
      ierr = TaoSetType(_tao, "blmvm" );
    else if (method == "gpcg")
      ierr = TaoSetType(_tao, "gpcg" );
    else if (method == "bqpip")
      ierr = TaoSetType(_tao, "bqpip");
    else
    {
      dolfin_error("TAOLinearBoundSolver.cpp",
                   "set solver for TAO solver",
                   "Unknown solver type (\"%s\")", method.c_str());
      ierr = 0; // Make compiler happy about uninitialized variable
    }
  }
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetType");
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_ksp(std::string ksp_type)
{
  PetscErrorCode ierr;

  // Set ksp type
  if (ksp_type != "default")
  {
    dolfin_assert(_tao);
    KSP ksp;
    ierr = TaoGetKSP(_tao, &ksp);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetKSP");
    if (ksp)
    {
      ierr = KSPSetType(ksp, _ksp_methods.find(ksp_type)->second);
      if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetType");
    }
    else
    {
      log(WARNING, "The selected tao solver does not allow to set a specific "\
      "Krylov solver. Option %s is ignored", ksp_type.c_str());
    }
  }
}
//-----------------------------------------------------------------------------
Tao TAOLinearBoundSolver::tao() const
{
  return _tao;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const PETScMatrix> TAOLinearBoundSolver::get_matrix() const
{
  return _matA;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const PETScVector> TAOLinearBoundSolver::get_vector() const
{
  return _b;
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::read_parameters()
{
  dolfin_assert(_tao);
  PetscErrorCode ierr;

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
  if (parameters["maximum_iterations"].is_set())
  {
    int maxits = parameters["maximum_iterations"];
    ierr = TaoSetMaximumIterations(_tao, maxits);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetMaximumIterations");
  }

  // Set ksp_options
  set_ksp_options();
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::init(const std::string& method)
{
  PetscErrorCode ierr;

  // Check that nobody else shares this solver
  if (_tao)
  {
    ierr = TaoDestroy(&_tao);
    if (ierr != 0) petsc_error(ierr, __FILE__, "TaoDestroy");
  }

  // Set up solver environment
  ierr = TaoCreate(PETSC_COMM_WORLD, &_tao);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoCreate");

  // Set tao solver
  set_solver(method);
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_ksp_options()
{
  dolfin_assert(_tao);
  PetscErrorCode ierr;

  KSP ksp;
  ierr = TaoGetKSP(_tao, &ksp);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetKSP");

  if (ksp)
  {
    Parameters krylov_parameters = parameters("krylov_solver");

    // Non-zero initial guess
    bool nonzero_guess = false;
    if (krylov_parameters["nonzero_initial_guess"].is_set())
      nonzero_guess = krylov_parameters["nonzero_initial_guess"];

    if (nonzero_guess)
      ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    else
      ierr = KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetInitialGuessNonzero");

    if (krylov_parameters["monitor_convergence"].is_set())
    {
      if (krylov_parameters["monitor_convergence"])
      {
        #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 6 && PETSC_VERSION_RELEASE == 1
        ierr = TaoSetMonitor(_tao, __TAOMonitor, this, NULL);
        if (ierr != 0) petsc_error(ierr, __FILE__, "TaoSetMonitor");
        #else
        PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));
        PetscViewerFormat format = PETSC_VIEWER_DEFAULT;
        PetscViewerAndFormat *vf;
        ierr = PetscViewerAndFormatCreate(viewer,format,&vf);
        ierr = KSPMonitorSet(ksp, (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*)) KSPMonitorTrueResidualNorm,
                             vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
        if (ierr != 0) petsc_error(ierr, __FILE__, "KSPMonitorSet");
        #endif
      }
    }

    // Set tolerances
    const double rtol = krylov_parameters["relative_tolerance"].is_set() ? (double)krylov_parameters["relative_tolerance"] : PETSC_DEFAULT;
    const double atol = krylov_parameters["absolute_tolerance"].is_set() ? (double)krylov_parameters["absolute_tolerance"] : PETSC_DEFAULT;
    const double dtol = krylov_parameters["divergence_limit"].is_set() ? (double)krylov_parameters["divergence_limit"] : PETSC_DEFAULT;
    const int max_ksp_it  = krylov_parameters["maximum_iterations"].is_set() ? (int)krylov_parameters["maximum_iterations"] : PETSC_DEFAULT;
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, max_ksp_it);
    if (ierr != 0) petsc_error(ierr, __FILE__, "KSPSetTolerances");

    // Set preconditioner
    if (_preconditioner && !_preconditioner_set)
    {
      PETScKrylovSolver dolfin_ksp(ksp);
      _preconditioner->set(dolfin_ksp);
      _preconditioner_set = true;
    }
  }
}
//-----------------------------------------------------------------------------
PetscErrorCode
TAOLinearBoundSolver::__TAOFormFunctionGradientQuadraticProblem(Tao tao,
                                                                Vec X,
                                                                PetscReal *ener,
                                                                Vec G,
                                                                void *ptr)
{
  PetscErrorCode ierr;
  PetscReal AXX, bX;

  dolfin_assert(ptr);
  const TAOLinearBoundSolver* solver = static_cast<TAOLinearBoundSolver*>(ptr);
  const PETScMatrix* A = solver->get_matrix().get();
  const PETScVector* b = solver->get_vector().get();
  dolfin_assert(A);
  dolfin_assert(b);

  // Calculate AX=A*X and store in G
  ierr = MatMult(A->mat(), X, G);
  if (ierr != 0) petsc_error(ierr, __FILE__, "MatMult");

  // Calculate AXX=A*X*X
  ierr = VecDot(G, X, &AXX);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecDot");

  // Calculate bX=b*X
  ierr = VecDot(b->vec(), X, &bX);
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecDot");

  // Calculate the functional value ener=1/2*A*X*X-b*X
  dolfin_assert(ener);
  *ener = 0.5*AXX-bX;

  // Calculate the gradient vector G=A*X-b
  ierr = VecAXPBY(G, -1.0, 1.0, b->vec());
  if (ierr != 0) petsc_error(ierr, __FILE__, "VecAXPBY");

  return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode
TAOLinearBoundSolver::__TAOFormHessianQuadraticProblem(Tao tao,
                                                       Vec X, Mat H,
                                                       Mat Hpre,
                                                       void *ptr)
{
  dolfin_assert(ptr);
  const TAOLinearBoundSolver* solver = static_cast<TAOLinearBoundSolver*>(ptr);
  const PETScMatrix* A = solver->get_matrix().get();
  dolfin_assert(A);

  // Set the hessian to the matrix A (quadratic problem)
  Mat Atmp = A->mat();
  H = Atmp;

  return 0;
}
//------------------------------------------------------------------------------
PetscErrorCode TAOLinearBoundSolver::__TAOMonitor(Tao tao, void *ctx)
{
  dolfin_assert(tao);
  PetscErrorCode ierr;
  PetscInt its;
  PetscReal f, gnorm, cnorm, xdiff;
  TaoConvergedReason reason;

  ierr = TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason);
  if (ierr != 0) petsc_error(ierr, __FILE__, "TaoGetSolutionStatus");
  if (!(its % 5))
  {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"iteration=%D\tf=%g\n", its, (double)f);
    if (ierr != 0) petsc_error(ierr, __FILE__, "PetscPrintf");
  }

  return 0;
}
//------------------------------------------------------------------------------

#endif
