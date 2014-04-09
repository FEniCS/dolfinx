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
//
// First added : 2012-12-03
// Last changed: 2013-04-04

#ifdef ENABLE_PETSC_TAO

#include <petsclog.h>

#include <dolfin/common/Timer.h>
#include <boost/assign/list_of.hpp>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
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

#include <dolfin/common/timing.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
// Mapping from ksp_method string to PETSc
const std::map<std::string, const KSPType> TAOLinearBoundSolver::_ksp_methods
  = boost::assign::map_list_of("default",    ""         )
                              ("cg",         KSPCG      )
                              ("gmres",      KSPGMRES   )
                              ("minres",     KSPMINRES  )
                              ("tfqmr",      KSPTFQMR   )
                              ("richardson", KSPRICHARDSON)
                              ("stcg",       KSPSTCG     )
                              ("bicgstab",   KSPBCGS    );
//-----------------------------------------------------------------------------
// Mapping from method string to description
const std::vector<std::pair<std::string, std::string> >
  TAOLinearBoundSolver::_methods_descr = boost::assign::pair_list_of
    ("default"  ,  "Default Tao method (tao_tron)"         )
    ("tron" ,  "Newton Trust Region method"            )
    ("bqpip",  "Interior Point Newton Algorithm"       )
    ("gpcg" ,  "Gradient Projection Conjugate Gradient")
    ("blmvm",  "Limited memory variable metric method" );
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
TAOLinearBoundSolver::methods()
{
  return TAOLinearBoundSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
TAOLinearBoundSolver::krylov_solvers()
{
  return PETScKrylovSolver::methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
TAOLinearBoundSolver::preconditioners()
{
  return PETScPreconditioner::preconditioners();
}
//-----------------------------------------------------------------------------
TAOLinearBoundSolver::TAOLinearBoundSolver(const std::string method,
                                           const std::string ksp_type,
                                           const std::string pc_type)
  : _tao(NULL), preconditioner(new PETScPreconditioner(pc_type)),
    preconditioner_set(false)
{
  // Set parameter values
  parameters = default_parameters();

  //Initialize the Tao solver
  init(method);

  //Set the PETSC KSP used by TAO
  set_ksp(ksp_type);

  // Some preconditioners may lead to errors because not compatible with TAO.
  if ((pc_type != "default") or (ksp_type != "default") or (method != "default"))
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
  this->A = A;
  this->b = b;
  dolfin_assert(this->A);
  dolfin_assert(this->b);
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

  // Check symmetry
  dolfin_assert(A1.size(0) == A1.size(1));

  // Set operators (A and b)
  std::shared_ptr<const PETScMatrix> _matA(&A1, NoDeleter());
  std::shared_ptr<const PETScVector> _b(&b1, NoDeleter());
  set_operators(_matA,_b);
  dolfin_assert(A->mat());
  //dolfin_assert(b->vec());

  // Set initial vector
  dolfin_assert(_tao);
  TaoSetInitialVector(_tao, x.vec());

  // Set the bound on the variables
  TaoSetVariableBounds(_tao, xl.vec(), xu.vec());

  // Set the user function, gradient, hessian evaluation routines and
  // data structures
  TaoSetObjectiveAndGradientRoutine(_tao,
                                    __TAOFormFunctionGradientQuadraticProblem,
                                    this);
  TaoSetHessianRoutine(_tao, A->mat(), A->mat(), __TAOFormHessianQuadraticProblem, this);

  // Set parameters from local parameters, including ksp parameters
  read_parameters();

  // Clear previous monitors
  TaoCancelMonitors(_tao);

  // Set the monitor
  if (parameters["monitor_convergence"])
    TaoSetMonitor(_tao, __TAOMonitor, this, PETSC_NULL);


  // Check for any tao command line options
  std::string prefix = std::string(parameters["options_prefix"]);
  if (prefix != "default")
  {
    // Make sure that the prefix has a '_' at the end if the user
    // didn't provide it
    char lastchar = *prefix.rbegin();
    if (lastchar != '_')
      prefix += "_";

    TaoSetOptionsPrefix(_tao, prefix.c_str());
  }
  TaoSetFromOptions(_tao);

  // Solve the bound constrained problem
  Timer timer("TAO solver");
  const char* tao_type;
  TaoGetType(_tao, &tao_type);
  log(PROGRESS, "Tao solver %s starting to solve %i x %i system", tao_type,
      A->size(0), A->size(1));

  TaoSolve(_tao);

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
      dolfin_error("TAOLinearBoundSolver.cpp",
                   "solve linear system using Tao solver",
                   "Solution failed to converge in %i iterations (TAO reason %d)",
                   num_iterations, reason);
    }
    else
    {
      log(WARNING,  "Tao solver %s failed to converge. Try a different TAO method,"\
	  " adjust some parameters", tao_type);
    }
  }

  return num_iterations;
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
void TAOLinearBoundSolver::set_ksp(std::string ksp_type)
{
  // Set ksp type
  if (ksp_type != "default")
  {
    dolfin_assert(_tao);
    KSP ksp;
    TaoGetKSP(_tao, &ksp);
    if (ksp)
      KSPSetType(ksp, _ksp_methods.find(ksp_type)->second);
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
  return A;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const PETScVector> TAOLinearBoundSolver::get_vector() const
{
  return b;
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::read_parameters()
{
  dolfin_assert(_tao);

  // Set tolerances
  TaoSetTolerances(_tao, parameters["function_absolute_tol"],
		   parameters["function_relative_tol"],
		   parameters["gradient_absolute_tol"],
		   parameters["gradient_relative_tol"],
		   parameters["gradient_t_tol"]);

  // Set TAO solver maximum iterations
  int maxits = parameters["maximum_iterations"];
  TaoSetMaximumIterations(_tao, maxits);

  // Set ksp_options
  set_ksp_options();
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::init(const std::string& method)
{
  // Check that nobody else shares this solver
  if (_tao)
    TaoDestroy(&_tao);

  // Set up solver environment
  TaoCreate(PETSC_COMM_WORLD, &_tao);

  // Set tao solver
  set_solver(method);
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_ksp_options()
{
  dolfin_assert(_tao);
  KSP ksp;
  TaoGetKSP(_tao, &ksp);
  if (ksp)
  {
    Parameters krylov_parameters = parameters("krylov_solver");

    // GMRES restart parameter
    const int gmres_restart = krylov_parameters("gmres")["restart"];
    KSPGMRESSetRestart(ksp, gmres_restart);

    // Non-zero initial guess
    const bool nonzero_guess = krylov_parameters["nonzero_initial_guess"];
    if (nonzero_guess)
      KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    else
      KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);

    if (krylov_parameters["monitor_convergence"])
      KSPMonitorSet(ksp, KSPMonitorTrueResidualNorm, 0, 0);

    // Set tolerances
    const int max_ksp_it = krylov_parameters["maximum_iterations"];
    KSPSetTolerances(ksp,
                     krylov_parameters["relative_tolerance"],
		     krylov_parameters["absolute_tolerance"],
		     krylov_parameters["divergence_limit"],
		     max_ksp_it);

    // Set preconditioner
    if (preconditioner && !preconditioner_set)
    {
      PETScKrylovSolver dolfin_ksp(ksp);
      preconditioner->set(dolfin_ksp);
      preconditioner_set = true;
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
   PetscReal AXX, bX;
   const TAOLinearBoundSolver* solver = static_cast<TAOLinearBoundSolver*>(ptr);
   const PETScMatrix* A = solver->get_matrix().get();
   const PETScVector* b = solver->get_vector().get();

   dolfin_assert(A);
   dolfin_assert(b);

   // Calculate AX=A*X and store in G
   MatMult(A->mat(), X, G);

   // Calculate AXX=A*X*X
   VecDot(G, X, &AXX);

   // Calculate bX=b*X
   VecDot(b->vec(), X, &bX);

   // Calculate the functional value ener=1/2*A*X*X-b*X
   *ener = 0.5*AXX-bX;

   // Calculate the gradient vector G=A*X-b
   VecAXPBY(G, -1.0, 1.0, b->vec());
   return 0;
}
//-----------------------------------------------------------------------------
PetscErrorCode
TAOLinearBoundSolver::__TAOFormHessianQuadraticProblem(Tao tao,
                                                       Vec X, Mat H,
                                                       Mat Hpre,
                                                       void *ptr)
{
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
  PetscInt its;
  PetscReal f, gnorm, cnorm, xdiff;
  TaoConvergedReason reason;
  TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason);
  if (!(its%5))
  {
    PetscPrintf(PETSC_COMM_WORLD,"iteration=%D\tf=%g\n",its,(double)f);
  }
  return 0;
}
//------------------------------------------------------------------------------
#endif
