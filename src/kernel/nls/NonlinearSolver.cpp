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
NonlinearSolver::NonlinearSolver() : _nlfunction(0), snes(0)
{
  // Initialize PETSc
  PETScManager::init();

  SNESCreate(PETSC_COMM_SELF, &snes);

}
//-----------------------------------------------------------------------------
NonlinearSolver::NonlinearSolver(NonlinearFunctional& nlfunction) : 
  _nlfunction(&nlfunction), snes(0)
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
void NonlinearSolver::solve(Vector& x)
{
  //FIXME
  // Initiate approximate solution vector
  Vector x0;
  x0.init(x.size());

  x0 = 1;  

  // RHS vector 
  Vector b;
  b.init(x.size());

  // Initiate matrix
  Matrix A;
  A.init(10,10);
  A(0,0) = 1.0; A(1,1) = 1.0; A(2,2) = 1.0; A(3,3) = 1.0; A(4,4) = 1.0; A(5,5) = 1.0; 
  A(6,6) = 1.0; A(7,7) = 1.0; A(8,8) = 1.0; A(9,9) = 1.0; 

  KSP ksp;
  PC  pc;
  
  // Get Krylov solver
  SNESGetKSP(snes, &ksp);

  // Set Krylov tolerances
  KSPSetTolerances(ksp, 1.0e-15, PETSC_DEFAULT, PETSC_DEFAULT, 10000);

  // Get Krylov preconditioner
  KSPGetPC(ksp, &pc);

  // Set preconditioner type
  PCSetType(pc, PCILU);

  // Set pointer to approximate solution vector
//  dolfin_error("Commented this one out, didn't compile for me. /Anders");
    SNESSetSolution(snes, x0.vec());

  // Set Jacobian Function
  SNESSetJacobian(snes, A.mat(), A.mat(), FormJacobian, _nlfunction);

  // Set RHS Function
  SNESSetFunction(snes, b.vec(), FormRHS, _nlfunction);

  SNESSetFromOptions(snes);

// Test Jacobian function
/*
  MatStructure  flg;
  Mat AA, BB;
  AA = A.mat();  
  BB = A.mat();  
  int test = SNESComputeJacobian(snes, x.vec(), &AA, &BB, &flg);
  cout << "After Jacobian test " << test << endl;  
*/


  int iter;
  // Get number of iterations
  SNESGetIterationNumber(snes, &iter);

  // Solve nonlinear problem
  SNESSolve(snes, PETSC_NULL, x.vec());
  cout << "Number of Newton iterations iterations: " << iter << endl;  


} 
//-----------------------------------------------------------------------------
int NonlinearSolver::FormRHS(SNES snes, Vec x, Vec f, void *prt)
{
  cout << "inside FormRHS " << endl;
  VecSetValue(f, 2, 1.2, INSERT_VALUES);

  return 0;
}
//-----------------------------------------------------------------------------
int NonlinearSolver::FormJacobian(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* ptr)
{
  cout << "inside FormJacobian " << endl;

  return 0;
}
//-----------------------------------------------------------------------------
