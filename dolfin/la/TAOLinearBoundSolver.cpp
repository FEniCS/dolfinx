//
// Created by Corrado Maurini 2012
//
// Last changed: 2012-12-03


#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin.h>
#include "petscdm.h"
#include "petscksp.h"
#include "petscvec.h" 
#include "petscmat.h"
#include "tao.h"
#include "taosolver.h"
#include "TAOLinearBoundSolver.h"


using namespace dolfin;

//-----------------------------------------------------------------------------
TAOLinearBoundSolver::TAOLinearBoundSolver(std::string method)
{
  // Set parameter values
  parameters = default_parameters();
  
  // Set up solver environment
  if (dolfin::MPI::num_processes() > 1)
    TaoCreate(PETSC_COMM_WORLD, &tao);
  else
    TaoCreate(PETSC_COMM_SELF, &tao);
  
  //TaoSetType(tao,"tao_tron");
  
}
//-----------------------------------------------------------------------------
TAOLinearBoundSolver::~TAOLinearBoundSolver()
{
  // Destroy solver environment
  if (tao)
  {
    TaoDestroy(&tao);
  }
}
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
  TaoSetInitialVector(tao,*x.vec()); 
  
  // Set the bound on the variables 
  TaoSetVariableBounds(tao,*xl.vec(),*xu.vec());
    
  // Set the user function, gradient, hessian evaluation routines and data structures //
  TaoSetObjectiveAndGradientRoutine(tao,TAOFormFunctionGradientQuadraticProblem,this);
  TaoSetHessianRoutine(tao,*A->mat(),*A->mat(),TAOFormHessianQuadraticProblem,this);
  
  // Set the linear solver used by TAO (parameters can be added to control this)
  KSP        ksp;
  TaoGetKSP(tao,&ksp); 
  if (ksp) 
  {                                         
    KSPSetType(ksp,KSPCG); 
  }
 
  // Set parameters from local parameters
  read_parameters();

  // Check for any tao command line options //
  TaoSetFromOptions(tao); 
  
  // Set the monitor
  if (parameters["monitor"])
  	TaoSetMonitor(tao,TAOMonitor,this,PETSC_NULL);  
  
  // Solve the bound constrained problem //
  TaoSolve(tao); 
  
  // Print the report on convergences and methods used
  if (parameters["report"])
	TaoView(tao,PETSC_VIEWER_STDOUT_WORLD); 
	
  // Check for convergence  
  TaoSolverTerminationReason reason;
  TaoGetTerminationReason(tao,&reason); 
  if (reason <= 0)
     PetscPrintf(PETSC_COMM_WORLD,"Try a different TAO method, adjust some parameters, or check the function evaluation routines\n");
     
}

void TAOLinearBoundSolver::set_solver(std::string solver)
{
  // Do nothing if default type is specified
  if (solver == "default")
    TaoSetType(tao,"tao_blmvm");

  // Choose solver
  if (solver == "tao_blmvm")
    TaoSetType(tao,"tao_blmvm");
  else if (solver == "tao_tron")
    TaoSetType(tao,"tao_tron");
  else if (solver == "tao_gpcg")
    TaoSetType(tao,"tao_gpcg");
  else
  {
    dolfin_error("TAOLinearBoundSolver.cpp",
                 "set solver for TAO solver",
                 "Unknown solver type (\"%s\")", solver.c_str());
  }  
}

void TAOLinearBoundSolver::read_parameters()
{
  set_solver(parameters["solver"]);
  set_tolerances(parameters["fatol"],parameters["frtol"],parameters["gatol"],parameters["grtol"],parameters["gttol"]);

}

//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_tolerances(double fatol, double frtol, double gatol, double grtol, double gttol)
{
  dolfin_assert(fatol > 0.0);
  dolfin_assert(frtol > 0.0);
  dolfin_assert(gatol > 0.0);
  dolfin_assert(grtol > 0.0);
  dolfin_assert(gttol > 0.0);

  TaoSetTolerances(tao, fatol, frtol, gatol, grtol, gttol);
}
//-----------------------------------------------------------------------------

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
