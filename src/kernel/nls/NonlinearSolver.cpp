// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005


#include <petscsnes.h>

#include <dolfin/PETScManager.h>
#include <dolfin/NonlinearSolver.h>
#include <dolfin/Parameter.h>
#include <dolfin/SettingsMacros.h>

using namespace dolfin;
//-----------------------------------------------------------------------------
NonlinearSolver::NonlinearSolver() : snes(0)
{
  // Initialize PETSc
  PETScManager::init();

  SNESCreate(PETSC_COMM_SELF, &snes);

}
//-----------------------------------------------------------------------------
NonlinearSolver::~NonlinearSolver()
{
  if( snes ) SNESDestroy(snes); 
}
//-----------------------------------------------------------------------------
dolfin::uint NonlinearSolver::solve(NonlinearFunction& nonlinear_function, Vector& x)
{

  Matrix A;
  Vector b, r;

  // Initialize global RHS vector, Jacobian matrix, solution vector and approximate 
  // solution vector
  const uint M  = nonlinear_function.size();
  const uint nz = nonlinear_function.nzsize();

  A.init(M, M, nz);
  b.init(M);
  x.init(M);
  r.init(M);
  A = 0.0; b = 0.0; r= 0.0;

  nonlinear_function.setJ(A);
  nonlinear_function.setF(b);

  // Set constant r (F(u) = r) (used for setting inhomeneous Dirichlet BC)
//  SNESSetRhs(snes, r.vec());


/*  
    Two options here to implemented.
    1: Compute F(u) and J at same time (more efficient for full Newton approach)
    2: Compute F(u) and J separately. Can then use modified Newton methods  
*/

  // 1: Compute nonlinear function and Jacobian function at same time 
  SNESSetFunction(snes, b.vec(), formSystem, &nonlinear_function);
  SNESSetJacobian(snes, A.mat(), A.mat(), formDummy, &nonlinear_function);


  // 2: Set nonlinear function and Jacobian function separately
  //SNESSetFunction(snes, b.vec(), formRHS, &nonlinear_function);
  //SNESSetJacobian(snes, A.mat(), A.mat(), formJacobian, &nonlinear_function);


  // Set options for linear solver
  KSP ksp;   
  SNESGetKSP(snes, &ksp);
  KSPSetTolerances(ksp, 1.0e-15, PETSC_DEFAULT, PETSC_DEFAULT, 10000);

  // Set preconditioner type
  PC  pc; 
  KSPGetPC(ksp, &pc);
//  PCSetType(pc, PCILU);
//  PCSetType(pc, PCSOR);
//  PCSetType(pc, PCJACOBI);
//  PCSetType(pc, PCICC);

  // Set monitor function
  SNESSetMonitor(snes, &monitor, PETSC_NULL, PETSC_NULL);  

  // Get Newton parameters from DOLFIN settings
  real rtol = dolfin_get("NLS Newton relative convergence tolerance");
  real stol = dolfin_get("NLS Newton successive convergence tolerance");
  real atol = dolfin_get("NLS Newton absolute convergence tolerance");
  int maxit = dolfin_get("NLS Newton maximum iterations");
  int maxf  = dolfin_get("NLS Newton maximum function evaluations");

//  SNESGetTolerances(snes, &atol, &rtol, &stol, &maxit, &maxf);

  dolfin_info("Newton solver tolerances (relative, successive, absolute) = (%.1e, %.1e, %.1e,).",
      rtol, stol, atol);
  dolfin_info("Newton solver maximum iterations = %d", maxit);
  
  // Set Netwon solver parameters
  SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf);

  // Set nonlinear solver parameters
  SNESSetFromOptions(snes);

  // Solve nonlinear problem
  SNESSolve(snes, PETSC_NULL, x.vec());

  // Report number of Newton iterations
  int iterations;
  SNESGetIterationNumber(snes, &iterations);
  dolfin_info("Newton solver finished in %d iterations.", iterations);

  return 0;
} 
//-----------------------------------------------------------------------------
void NonlinearSolver::setMaxiter(int maxiter)
{
  dolfin_set("NLS Newton maximum iterations", maxiter);
  dolfin_info("Maximum number of Newton iterations: %d.",maxiter);
}
//-----------------------------------------------------------------------------
void NonlinearSolver::setRtol(double rtol)
{
  dolfin_set("NLS Newton relative convergence tolerance", rtol);
  dolfin_info("Relative increment tolerance for Newton solver: %e.", rtol);
}
//-----------------------------------------------------------------------------
void NonlinearSolver::setStol(double stol)
{
  dolfin_set("NLS Newton successive convergence tolerance", stol);
  dolfin_info("Successive increment tolerance for Newton solver: %e.", stol);
}
//-----------------------------------------------------------------------------
void NonlinearSolver::setAtol(double atol)
{
  dolfin_set("NLS Newton successive convergence tolerance", atol);
  dolfin_info("Absolute increment tolerance for Newton solver: %e.", atol);
}
//-----------------------------------------------------------------------------
int NonlinearSolver::formRHS(SNES snes, Vec x, Vec f, void *nlProblem)
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
int NonlinearSolver::formJacobian(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlProblem)
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
int NonlinearSolver::formSystem(SNES snes, Vec x, Vec f, void *nlProblem)
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
int NonlinearSolver::formDummy(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlProblem)
{
  // Dummy function for computing Jacobian to trick PETSc. Do nothing.
  return 0;
}
//-----------------------------------------------------------------------------
int NonlinearSolver::monitor(SNES snes, int iter, real fnorm , void* dummy)
{
  dolfin_info("Iteration number = %d, residual norm = %e.", iter, fnorm);
  return 0;
}
//-----------------------------------------------------------------------------
