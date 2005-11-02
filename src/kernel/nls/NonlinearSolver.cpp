// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005


#include <petscsnes.h>

#include <dolfin/PETScManager.h>
#include <dolfin/NonlinearSolver.h>

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
  const uint nz = 10;

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
  PCSetType(pc, PCILU);


  // Set monitor
  SNESSetRatioMonitor(snes);  

  // Set nonlinear solver parameters
  SNESSetFromOptions(snes);

  // Solve nonlinear problem
  SNESSolve(snes, PETSC_NULL, x.vec());

  // Report number of Newton iterations
  int iterations;
  SNESGetIterationNumber(snes, &iterations);
  dolfin_info("Nonlinear solver converged in %d iterations.", iterations);

  return 0;
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
  // Dummy function for computing Jacobian. Do nothing.
  return 0;
}
//-----------------------------------------------------------------------------
