//
// Created by Corrado Maurini 2012
//
// First added:  2012-12-03
#ifdef HAS_PETSC
#ifdef HAS_TAO

#include <petsclog.h>

#include <boost/assign/list_of.hpp>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScKrylovSolver.h"
#include "PETScPreconditioner.h"
#include "TAOLinearBoundSolver.h"
#include "petscksp.h"
#include "petscvec.h" 
#include "petscmat.h"
#include "tao.h"
#include "taosolver.h"

#include <dolfin/common/timing.h>


using namespace dolfin;

// Utility function
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
//-----------------------------------------------------------------------------
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
// Mapping from method string to description
const std::vector<std::pair<std::string, std::string> >
  TAOLinearBoundSolver::_methods_descr = boost::assign::pair_list_of
    ("default"  ,  "Default Tao method (tao_tron)")
    ("tao_blmvm",  "Limited memory variable metric method")
    ("tao_tron" ,  "Newton Trust Region method"     )
    ("tao bqpip",  "Interior Point Newton Algorithm")
    ("tao_gpcg" ,  "Gradient Projection Conjugate Gradient");
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
TAOLinearBoundSolver::TAOLinearBoundSolver(std::string method, 
                                           std::string ksp_type, 
                                           std::string pc_type) 
        : preconditioner(new PETScPreconditioner(pc_type)), preconditioner_set(false) 
{
  // Set parameter values
  parameters = default_parameters();
  
  //Initialize the Tao solver
  init(method);
    
  // Set ksp type
  if (ksp_type != "default")
  {
    KSP ksp; 
    TaoGetKSP(*_tao, &ksp);
    KSPSetType(ksp, _ksp_methods.find(ksp_type)->second);   
   }
   
   // Some preconditioners may lead to errors because not compatible with TAO.
   if ((pc_type != "default") or (ksp_type != "default") or (method != "default"))
   {
   PetscPrintf(PETSC_COMM_WORLD,"Warning: some preconditioners may be not applicable to TAO and generate errors. Needs testing.");
   }
}
//-----------------------------------------------------------------------------
TAOLinearBoundSolver::~TAOLinearBoundSolver()
{
  // Do nothing
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
  dolfin_assert(*_tao)
  TaoSetInitialVector(*_tao,*x.vec()); 
  
  // Set the bound on the variables 
  TaoSetVariableBounds(*_tao,*xl.vec(),*xu.vec());
    
  // Set the user function, gradient, hessian evaluation routines and data structures
  TaoSetObjectiveAndGradientRoutine(*_tao,__TAOFormFunctionGradientQuadraticProblem,this);
  TaoSetHessianRoutine(*_tao,*A->mat(),*A->mat(),__TAOFormHessianQuadraticProblem,this);
  
  // Set parameters from local parameters, including ksp parameters
  read_parameters(); 

  // Check for any tao command line options
  TaoSetFromOptions(*_tao); 
  
  // Set the monitor
  if (parameters["monitor_convergence"])
  {
  	TaoSetMonitor(*_tao,__TAOMonitor,this,PETSC_NULL);  
  }
  
  // Solve the bound constrained problem 
  PetscPrintf(MPI_COMM_WORLD, "Tao solver starting to solve %i x %i system\n",
        A->size(0), A->size(1));
  
  TaoSolve(*_tao); 
  
  // Print the report on convergences and methods used
  if (parameters["report"])
	TaoView(*_tao,PETSC_VIEWER_STDOUT_WORLD); 
    
  // Check for convergence  
  TaoSolverTerminationReason reason;
  TaoGetTerminationReason(*_tao,&reason); 

  // Get the number of iterations
  int num_iterations = 0;
  TaoGetMaximumIterations(*_tao, &num_iterations);
  
  // Report number of iterations
  if (reason >= 0)
  {
    PetscPrintf(MPI_COMM_WORLD, "Tao solver converged\n");
  }
  else
  {
    PetscPrintf(MPI_COMM_WORLD, "Tao solver failed to converge. Try a different TAO method, adjust some parameters\n");
  }
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_solver(const std::string method)
{
  // Do nothing if default type is specified
  if (method == "default")
    TaoSetType(*_tao,"tao_tron");
  else
  {  
    // Choose solver
    if (method == "tao_blmvm") 
        TaoSetType(*_tao,method.c_str());
    else if (method == "tao_tron")
        TaoSetType(*_tao,method.c_str() );
    else if (method == "tao_gpcg")
        TaoSetType(*_tao, method.c_str() );
    else if (method == "tao_bqpip")
        TaoSetType(*_tao, method.c_str() );
    else
        dolfin_error("TAOLinearBoundSolver.cpp",
                  "set solver for TAO solver",
                   "Unknown solver type (\"%s\")", method.c_str());
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
  // Set tao method
  TaoSetTolerances(*_tao, 
                            parameters["function_absolute_tol"],
                            parameters["function_relative_tol"],
                            parameters["gradient_absolute_tol"],
                            parameters["gradient_relative_tol"],
                            parameters["gradient_t_tol"]
                   );
  // Set ksp solver parameters
  set_ksp_options();
}
//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::init(const std::string& method)
{
  // Check that nobody else shares this solver
  if (_tao && !_tao.unique()) 
  {
    dolfin_error("TAOLinearBoundSolver.cpp",
                 "initialize TAO solver",
                 "More than one object points to the underlying PETSc object");
  }

  // Create new TAO object
  _tao.reset(new TaoSolver, TAODeleter());

  // Set up solver environment
  if (MPI::num_processes() > 1)
    TaoCreate(PETSC_COMM_WORLD, _tao.get());
  else
    TaoCreate(PETSC_COMM_SELF, _tao.get());
  
  // Set tao solver
  set_solver(method);
}    

//-----------------------------------------------------------------------------
void TAOLinearBoundSolver::set_ksp_options()
{ 
  KSP ksp; 
  TaoGetKSP(*_tao, &ksp);
  Parameters krylov_parameters = parameters("krylov_solver");
  // GMRES restart parameter
  KSPGMRESSetRestart(ksp,krylov_parameters("gmres")["restart"]);

  // Non-zero initial guess
  const bool nonzero_guess = krylov_parameters["nonzero_initial_guess"];
  if (nonzero_guess)
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  else
    KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);

  if (krylov_parameters["monitor_convergence"])
    KSPMonitorSet(ksp, KSPMonitorTrueResidualNorm, 0, 0);
    
  // Set tolerances
  KSPSetTolerances(ksp,
                   krylov_parameters["relative_tolerance"],
                   krylov_parameters["absolute_tolerance"],
                   krylov_parameters["divergence_limit"],
                   krylov_parameters["maximum_iterations"]);
  
  // Set preconditioner
  if (preconditioner && !preconditioner_set)
   {    
     PETScKrylovSolver dolfin_ksp(reference_to_no_delete_pointer(ksp));
     preconditioner->set(dolfin_ksp);
     preconditioner_set = true;
   }
                   
}
//-------------------------------------------------------------------------------------------
namespace dolfin
{
//-------------------------------------------------------------------------------------------
// Auxiliary non-member functions required by Tao to assemble gradient and hessian of the goal functional 
// This is implemented only for a quadratic functional (the hessian is constant)
// The should should be more properly 
//-------------------------------------------------------------------------------------------
PetscErrorCode __TAOFormFunctionGradientQuadraticProblem(TaoSolver tao, Vec X, PetscReal *ener, Vec G, void *ptr)
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
PetscErrorCode __TAOFormHessianQuadraticProblem(TaoSolver tao,Vec X,Mat *H, Mat *Hpre, MatStructure *flg, void *ptr)
{

   TAOLinearBoundSolver* 		 solver=(TAOLinearBoundSolver*)ptr;
   
   dolfin_assert(solver.A->mat());
   
   // Set the hessian to the matrix A (quadratic problem)
   H = solver->A->mat().get(); 
   //*H = *(solver->A->mat());
   return 0;
}

//-------------------------------------------------------------------------------------------
// Auxiliary function to set the output of the TaoMonitor 
//-------------------------------------------------------------------------------------------
PetscErrorCode __TAOMonitor(TaoSolver tao, void *ctx)
{
  PetscInt its;
  PetscReal f,gnorm,cnorm,xdiff;
  TaoSolverTerminationReason reason;
  TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason);
  PetscPrintf(PETSC_COMM_WORLD,"TAO iteration = %3D \tf=%10G\tgnorm=%-10G\tcnorm=%-10G\txdiff=%G\n",its,f,gnorm,cnorm,xdiff);
}
}
#endif
#endif

