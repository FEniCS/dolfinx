//
// Created by Corrado Maurini 2012
//
// Last changed: 2012-12-03
#ifdef HAS_PETSC
#ifdef HAS_TAO

#include <petsclog.h>

#include <boost/assign/list_of.hpp>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin.h>
#include <dolfin/parameter/GlobalParameters.h>
//#include "petscdm.h"
#include "petscksp.h"
#include "petscvec.h" 
#include "petscmat.h"
#include "tao.h"
#include "taosolver.h"
#include "TAOLinearBoundSolver.h"
#include "PETScKrylovSolver.h"

#include <dolfin/common/timing.h>


using namespace dolfin;

// Utility functions
namespace dolfin
{
  class TAODeleter
  {
  public:
    void operator() (TaoSolver* _tao)
    {
      if (_tao)
        TaoDestroy(_tao);
      delete _tao;
    }
  };
}

// Mapping from ksp_method string to PETSc
const std::map<std::string, const KSPType> TAOLinearBoundSolver::_ksp_methods
  = boost::assign::map_list_of("default",  "")
                              ("cg",         KSPCG)
                              ("gmres",      KSPGMRES)
                              ("minres",     KSPMINRES)
                              ("tfqmr",      KSPTFQMR)
                              ("richardson", KSPRICHARDSON)
                              ("bicgstab",   KSPBCGS);

//-----------------------------------------------------------------------------
TAOLinearBoundSolver::TAOLinearBoundSolver(std::string method)
{
  // Set parameter values
  parameters = default_parameters();
  
  // Set up solver environment
  //if (dolfin::MPI::num_processes() > 1)
  //  TaoCreate(PETSC_COMM_WORLD, _tao.get());
  //else
  //  TaoCreate(PETSC_COMM_SELF, _tao.get());

  init(method);  
}
//-----------------------------------------------------------------------------
TAOLinearBoundSolver::~TAOLinearBoundSolver()
{
  // Do nothing
}
//{
//  // Destroy solver environment
//  if (tao)
//  {
//    TaoDestroy(&tao);
//    
//  }
//}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_operators(const boost::shared_ptr<const GenericMatrix> A,
                                      const boost::shared_ptr<const GenericVector> b)
{
  boost::shared_ptr<const PETScMatrix> _A = GenericTensor::down_cast<const PETScMatrix>(A);
  boost::shared_ptr<const PETScVector> _b = GenericTensor::down_cast<const PETScVector>(b);
  set_operators(_A, _b);
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_operators(const boost::shared_ptr<const PETScMatrix> A,
                                      const boost::shared_ptr<const PETScVector> b)
	{
  		this->A = A;
  		this->b = b;
  		dolfin_assert(this->A);
  		dolfin_assert(this->b);
	}
//-----------------------------------------------------------------------------
dolfin::uint TAOLinearBoundSolver::solve(const GenericMatrix& A1, GenericVector& x, const GenericVector& b1, const GenericVector& xl, const GenericVector& xu)
{
  return solve(A1.down_cast<PETScMatrix>(), x.down_cast<PETScVector>(), b1.down_cast<PETScVector>(), xl.down_cast<PETScVector>(), xu.down_cast<PETScVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint TAOLinearBoundSolver::solve(const PETScMatrix& A1, PETScVector& x, const PETScVector& b1, const PETScVector& xl, const PETScVector& xu)
{
 
  // Check symmetry
  dolfin_assert(A->size(0) == A->size(1));
  
  // Set operators (A and b)
  boost::shared_ptr<const PETScMatrix> _A(&A1, NoDeleter());
  boost::shared_ptr<const PETScVector> _b(&b1, NoDeleter());
  set_operators(_A,_b);
  dolfin_assert(A->mat());
  dolfin_assert(b->vec());
  
  // Set initial vector  
  dolfin_assert(*_tao)
  TaoSetInitialVector(*_tao,*x.vec()); 
  
  // Set the bound on the variables 
  TaoSetVariableBounds(*_tao,*xl.vec(),*xu.vec());
    
  // Set the user function, gradient, hessian evaluation routines and data structures //
  TaoSetObjectiveAndGradientRoutine(*_tao,TAOFormFunctionGradientQuadraticProblem,this);
  TaoSetHessianRoutine(*_tao,*A->mat(),*A->mat(),TAOFormHessianQuadraticProblem,this);
  
  // Set parameters from local parameters
  read_parameters(); 

  // Check for any tao command line options
  TaoSetFromOptions(*_tao); 
  
  // Set the monitor
  if (parameters["monitor"])
  	TaoSetMonitor(*_tao,TAOMonitor,this,PETSC_NULL);  
  
  // Solve the bound constrained problem //
  TaoSolve(*_tao); 
  
  // Print the report on convergences and methods used
  if (parameters["report"])
	TaoView(*_tao,PETSC_VIEWER_STDOUT_WORLD); 
    
  // Check for convergence  
  TaoSolverTerminationReason reason;
  TaoGetTerminationReason(*_tao,&reason); 
  if (reason <= 0)
     PetscPrintf(PETSC_COMM_WORLD,"Try a different TAO method, adjust some parameters, or check the function evaluation routines\n");
     
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_solver(std::string method)
{
  // Do nothing if default type is specified
  if (method != "default")
  {  
    // Choose solver
    if (method == "tao_blmvm")
        TaoSetType(*_tao,"tao_blmvm");
    else if (method == "tao_tron")
        TaoSetType(*_tao,"tao_tron");
    else if (method == "tao_gpcg")
        TaoSetType(*_tao,"tao_gpcg");
    else
    {
        dolfin_error("TAOLinearBoundSolver.cpp",
                  "set solver for TAO solver",
                   "Unknown solver type (\"%s\")", method.c_str());
    }
  }  
}
//-----------------------------------------------------------------------------
boost::shared_ptr<TaoSolver> TAOLinearBoundSolver::tao() const
{
  return _tao;
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::read_parameters()
{
  set_solver(parameters["method"]);
  TaoSetTolerances(*_tao, parameters["fatol"],parameters["frtol"],parameters["gatol"],parameters["grtol"],parameters["gttol"]);
  set_ksp_options();
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::init(const std::string& method)
{
  // Check that nobody else shares this solver
  if (_tao && !_tao.unique())
  {
    dolfin_error("TAOLinearBoundSolver.cpp",
                 "initialize PETSc Krylov solver",
                 "More than one object points to the underlying PETSc object");
  }

  // Create new TAO object
  _tao.reset(new TaoSolver, TAODeleter());

  // Set up solver environment
  if (MPI::num_processes() > 1)
    TaoCreate(PETSC_COMM_WORLD, _tao.get());
  else
    TaoCreate(PETSC_COMM_SELF, _tao.get());
  
  // Set solver type
  set_solver(method);
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_ksp_options()
{
  // Get TAO KSP solver
  KSP tao_ksp;
  TaoGetKSP(*_tao,&tao_ksp);
  
  // Set Type
  const std::string& method= parameters("krylov_solver")["method"];
  if (method != "default")
    KSPSetType(tao_ksp, _ksp_methods.find(method)->second);
  
  // GMRES restart parameter
  KSPGMRESSetRestart(tao_ksp, parameters("krylov_solver")["gmres_restart"]);

  // Non-zero initial guess
  const bool nonzero_guess = parameters("krylov_solver")["nonzero_initial_guess"];
  if (nonzero_guess)
    KSPSetInitialGuessNonzero(tao_ksp, PETSC_TRUE);
  else
    KSPSetInitialGuessNonzero(tao_ksp, PETSC_FALSE);

  if (parameters("krylov_solver")["monitor_convergence"])
    KSPMonitorSet(tao_ksp, KSPMonitorTrueResidualNorm, 0, 0);
    
  // Set tolerances
  KSPSetTolerances(tao_ksp,
                   parameters("krylov_solver")["relative_tolerance"],
                   parameters("krylov_solver")["absolute_tolerance"],
                   parameters("krylov_solver")["divergence_limit"],
                   parameters("krylov_solver")["maximum_iterations"]);
}
//-----------------------------------------------------------------------------
//
//
//-----------------------------------------------------------------------------
PetscErrorCode TAOFormFunctionGradientQuadraticProblem(TaoSolver tao, Vec X, PetscReal *ener, Vec G, void *ptr)
{ 
   PetscReal 				 AXX, bX;
   TAOLinearBoundSolver* 	 solver=(TAOLinearBoundSolver*)ptr;
   
   dolfin_assert(solver.A->mat());
   dolfin_assert(solver.b->vec());
   
   // Calculate AX=A*X and store in G
   MatMult(*(solver->A->mat()),X,G); 
   // Calculate AXX=A*X*X
   VecDot(G,X,&AXX);
   // Calculate bX=b*X
   VecDot(*solver->b->vec(),X,&bX); 
   // Calculate the functional value ener=1/2*A*X*X-b*X
   *ener=0.5*AXX-bX;
   // Calculate the gradient vector G=A*X-b
   VecAXPBY(G,-1.0,1.0,*solver->b->vec()); 
   return 0;
}

//-----------------------------------------------------------------------------
PetscErrorCode TAOFormHessianQuadraticProblem(TaoSolver tao,Vec X,Mat *H, Mat *Hpre, MatStructure *flg, void *ptr)
{

   TAOLinearBoundSolver* 		 solver=(TAOLinearBoundSolver*)ptr;
   
   dolfin_assert(solver.A->mat());
   
   // Set the hessian to the matrix A (quadratic problem)
   *H = *(solver->A->mat());
   return 0;
}

//-----------------------------------------------------------------------------
PetscErrorCode TAOMonitor(TaoSolver tao, void *ctx)
{
  PetscInt its;
  PetscReal f,gnorm,cnorm,xdiff;
  TaoSolverTerminationReason reason;
  TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason);
  PetscPrintf(PETSC_COMM_WORLD,"iteration=%D\tf=%-10G\tgnorm=%-10G\tcnorm=%-10G\txdiff=%G\n",its,f,gnorm,cnorm,xdiff);
}

#endif
#endif

