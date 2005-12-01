// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005


#include <petscsnes.h>

#include <dolfin/PETScManager.h>
#include <dolfin/NewtonSolver.h>
#include <dolfin/Parameter.h>
#include <dolfin/SettingsMacros.h>

using namespace dolfin;
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver() : _nonlinear_function(0), _A(0), _b(0), _x(0)
{
  // Initialize PETSc
  PETScManager::init();

  SNESCreate(PETSC_COMM_SELF, &snes);

}
//-----------------------------------------------------------------------------
NewtonSolver::NewtonSolver(NonlinearFunction& nonlinear_function) 
    : _A(0), _b(0), _x(0)
{
  // Initialize PETSc
  PETScManager::init();

  SNESCreate(PETSC_COMM_SELF, &snes);
  _nonlinear_function = &nonlinear_function;

}
//-----------------------------------------------------------------------------
NewtonSolver::~NewtonSolver()
{
  if( snes ) SNESDestroy(snes); 
}
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve(NonlinearFunction& nonlinear_function, Vector& x)
{
  Matrix A;
  Vector b;

  _nonlinear_function = &nonlinear_function;

  // Allocate matrix and vectors
  init(A, b, x);

  // Set nonlinear solver parameters
  setParameters();

  // Solve
  solve();

  return 0;
} 
//-----------------------------------------------------------------------------
SNES NewtonSolver::solver()
{
  return snes;
} 
//-----------------------------------------------------------------------------
dolfin::uint NewtonSolver::solve()
{
  // Set solver type
  std::string solver_form = dolfin_get("NLS form");
  if(solver_form.c_str() == "concurrent")   // Compute F(u) and J at same time
  {
    SNESSetFunction(snes, _b->vec(), formSystem, _nonlinear_function);
    SNESSetJacobian(snes, _A->mat(), _A->mat(), formDummy, _nonlinear_function);
  }
  else  // Compute F(u) and J separately
  {
    SNESSetFunction(snes, _b->vec(), formRHS, _nonlinear_function);
    SNESSetJacobian(snes, _A->mat(), _A->mat(), formJacobian, _nonlinear_function);
  }
  
  SNESSetFromOptions(snes);

  // Solve nonlinear problem
  SNESSolve(snes, PETSC_NULL, _x->vec());

  // Report number of Newton iterations and solver iterations
  int iterations;
  SNESGetIterationNumber(snes, &iterations);
  int linear_iterations;
  SNESGetNumberLinearIterations(snes, &linear_iterations);

  dolfin_info("   Newton solver finished in %d iterations.", iterations);
  dolfin_info("   Total number of linear solver iterations: %d.", linear_iterations);

  return 0;
} 
//-----------------------------------------------------------------------------
void NewtonSolver::setParameters()
{

  //FIXME: use options database for Krylov solver to set linear solver parameters

  // Set options for linear solver and preconditioner
  KSP ksp;   
  SNESGetKSP(snes, &ksp);
  KSPSetTolerances(ksp, 1.0e-15, PETSC_DEFAULT, PETSC_DEFAULT, 10000);

  PC  pc; 
  KSPGetPC(ksp, &pc);
//  PCSetType(pc, PCILU);

  // Set monitor function for Newton iterations
  SNESSetMonitor(snes, &monitor, PETSC_NULL, PETSC_NULL);  

  // Get Newton parameters from DOLFIN settings
  real rtol = dolfin_get("NLS Newton relative convergence tolerance");
  real stol = dolfin_get("NLS Newton successive convergence tolerance");
  real atol = dolfin_get("NLS Newton absolute convergence tolerance");
  int maxit = dolfin_get("NLS Newton maximum iterations");
  int maxf  = dolfin_get("NLS Newton maximum function evaluations");

  dolfin_info("Newton solver tolerances (relative, successive, absolute) = (%.1e, %.1e, %.1e,).",
      rtol, stol, atol);
  dolfin_info("Newton solver maximum iterations = %d", maxit);
  
  // Set Newton solver tolerances
  SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf);

  // Set solver type
  std::string solver_type = dolfin_get("NLS type");
  if(solver_type  == "line search") // line search
  {
    SNESSetType(snes, SNESLS); 
  }
  else if(solver_type == "trust region") // trust region 
  {
    SNESSetType(snes, SNESTR); 
  }  

} 
//-----------------------------------------------------------------------------
void NewtonSolver::init(Matrix& A, Vector& b, Vector& x)
{
  cout << "Initialising " << endl;

  if( !_nonlinear_function )
     dolfin_error("Newton solver has not been correctly initialised");

  // Initialize global RHS vector, Jacobian matrix and solution vector if
  // necessary
  const uint M  = _nonlinear_function->size();
  const uint nz = _nonlinear_function->nzsize();

//  if(A.size(0) != M)
    A.init(M, M, nz);

//  if(b.size() != M)
    b.init(M);

//  if(x.size() != M)
    x.init(M);
  
  A = 0.0; 
  b = 0.0; 
  
  // Set intial guess to zero
  x = 0.0;

  // Set pointers to Jacobian matrix and RHS vector
  _A = &A;
  _b = &b;
  _x = &x;

  // Set pointers to Jacobian matrix and RHS vector
  _nonlinear_function->setJ(A);
  _nonlinear_function->setF(b);

} 
//-----------------------------------------------------------------------------
int NewtonSolver::getIteration(SNES snes)
{
  int iteration;
  SNESGetIterationNumber(snes, &iteration);
  return iteration; 
}
//-----------------------------------------------------------------------------
void NewtonSolver::setType(std::string solver_type)
{
  dolfin_set("NLS type", solver_type.c_str());
  dolfin_info("Nonlinear solver type: %s.", solver_type.c_str());
}
//-----------------------------------------------------------------------------
void NewtonSolver::setMaxiter(int maxiter)
{
  dolfin_set("NLS Newton maximum iterations", maxiter);
  dolfin_info("Maximum number of Newton iterations: %d.",maxiter);
}
//-----------------------------------------------------------------------------
void NewtonSolver::setRtol(double rtol)
{
  dolfin_set("NLS Newton relative convergence tolerance", rtol);
  dolfin_info("Relative increment tolerance for Newton solver: %e.", rtol);
}
//-----------------------------------------------------------------------------
void NewtonSolver::setStol(double stol)
{
  dolfin_set("NLS Newton successive convergence tolerance", stol);
  dolfin_info("Successive increment tolerance for Newton solver: %e.", stol);
}
//-----------------------------------------------------------------------------
void NewtonSolver::setAtol(double atol)
{
  dolfin_set("NLS Newton absolute convergence tolerance", atol);
  dolfin_info("Absolute increment tolerance for Newton solver: %e.", atol);
}
//-----------------------------------------------------------------------------
int NewtonSolver::formRHS(SNES snes, Vec x, Vec f, void *nlProblem)
{
  // Pointer to nonlinear function
  NonlinearFunction* nonlinear_function = (NonlinearFunction*)nlProblem;

  // Vector for F(u)
  Vector b(f);

  // Vector containing latest solution
  const Vector xsol(x);
  
  nonlinear_function->F(b, xsol);

  return 0;
}
//-----------------------------------------------------------------------------
int NewtonSolver::formJacobian(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlProblem)
{
  // Pointer to NonlinearFunction
  NonlinearFunction* nonlinear_function = (NonlinearFunction*)nlProblem;

  Matrix& A  = nonlinear_function->J();

  // Updated solution vector
  const Vector xsol(x);

  // Assemble Jacobian vector
  nonlinear_function->J(A, xsol);

  // Structure of returned matrix
  *flag = SAME_NONZERO_PATTERN;

  return 0;
}
//-----------------------------------------------------------------------------
int NewtonSolver::formSystem(SNES snes, Vec x, Vec f, void *nlProblem)
{
  // Pointer to nonlinear function
  NonlinearFunction* nonlinear_function = (NonlinearFunction*)nlProblem;

  Matrix& A  = nonlinear_function->J();

  // Vector for F(u)
  Vector b(f);

  // Updated solution vector
  const Vector xsol(x);
  
  nonlinear_function->form(A, b, xsol);

  return 0;
}
//-----------------------------------------------------------------------------
int NewtonSolver::formDummy(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlProblem)
{
  // Dummy function for computing Jacobian to trick PETSc. Do nothing.
  return 0;
}
//-----------------------------------------------------------------------------
int NewtonSolver::monitor(SNES snes, int iter, real fnorm , void* dummy)
{
  dolfin_info("   Iteration number = %d, residual norm = %e.", iter, fnorm);
  return 0;
}
//-----------------------------------------------------------------------------


