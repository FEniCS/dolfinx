// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005

#include <petscsnes.h>

#include <dolfin/PETScManager.h>
#include <dolfin/NonlinearSolver.h>
#include <dolfin/FEM.h>

using namespace dolfin;
//-----------------------------------------------------------------------------
NonlinearSolver::NonlinearSolver() : nlfunc(0), snes(0)
{
  // Initialize PETSc
  PETScManager::init();

  SNESCreate(PETSC_COMM_SELF, &snes);

}
//-----------------------------------------------------------------------------
NonlinearSolver::NonlinearSolver(NonlinearFunctional& nlfunction) : 
  nlfunc(&nlfunction), snes(0)
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
void NonlinearSolver::solve(Vector& x, NonlinearFunctional& nlfu)
{
  //FIXME
  // Initiate approximate solution vector
  Vector x0;
  x0.init(x.size());

  x0 = 1;  

  // RHS vector 
  Vector* b;
  b = nlfu._b;


  Matrix* A;
  A = nlfu._A;

//  b.init(x.size());
//  cout << "Display vector b " << endl;
//  b.disp();

  // Initiate matrix
//  Matrix A;
//  A.init(x.size(),x.size());
//  A(0,0) = 1.0; A(1,1) = 1.0; A(2,2) = 1.0; A(3,3) = 1.0; A(4,4) = 1.0; A(5,5) = 1.0; 
//  A(6,6) = 1.0; A(7,7) = 1.0; A(8,8) = 1.0; A(9,9) = 1.0; 

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

  /*
  // Set pointer to approximate solution vector
  SNESSetSolution(snes, x0.vec());
  */
  // Set RHS Function
  SNESSetFunction(snes, b->vec(), FormRHS, nlfunc);

  // Set Jacobian Function
  SNESSetJacobian(snes, A->mat(), A->mat(), FormJacobian, nlfunc);

// Test Jacobian function

//  MatStructure  flg;
//  Mat AA, BB;
//  AA = A.mat();  
//  BB = A.mat();  
//  int test = SNESComputeJacobian(snes, x.vec(), &AA, &BB, &flg);
//  cout << "After Jacobian test " << test << endl;  


  SNESSetFromOptions(snes);

  int iter;
  // Get number of iterations
  SNESGetIterationNumber(snes, &iter);

  // Solve nonlinear problem
  SNESSolve(snes, PETSC_NULL, x.vec());
  cout << "Number of Newton iterations iterations: " << iter << endl;  


} 
//-----------------------------------------------------------------------------
int NonlinearSolver::FormRHS(SNES snes, Vec x, Vec f, void *nlProblem)
{
  cout << "inside FormRHS " << endl;

  NonlinearFunctional* nlp = (NonlinearFunctional*)nlProblem;

  // Extract parts of the nonlinear problem
//  BilinearForm aa       = *(nlp->_a);
//  LinearForm L         = *nlp->_L;
//  Matrix A             = *nlp->_A;
//  Vector b             = *nlp->_b;
//  Vector x0            = *nlp->_x0;
//   Mesh* mesh            = (Mesh*)(nlp->_mesh);
//  BoundaryCondition bc = *nlp->_bc;

//  cout << "testing " << mesh->noNodes() <<  endl;

  // Update functions (user defined)
//  nlp->UpdateNonlinearFunction();

  // Initialise b vector with pointer to PETSc RHS vector f
//  Vector b(x);

  // Form RHS vector
  cout << "assemble " <<  endl;
//  FEM::assemble(a, L, A, b, mesh, bc);
  FEM::assemble(*nlp->_a, *nlp->_L, *nlp->_A, *nlp->_b, *nlp->_mesh, *nlp->_bc);
//  FEM::assemble(aa, *nlp->_L, *nlp->_A, *nlp->_b, *nlp->_mesh, *nlp->_bc);
  cout << "finish assemble " <<   endl;

  VecSetValue(f, 2, 1.2, INSERT_VALUES);

  cout << "finished inside FormRHS " << endl;

  return 0;
}
//-----------------------------------------------------------------------------
int NonlinearSolver::FormJacobian(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlProblem)
{
  cout << "inside FormJacobian " << endl;
  NonlinearFunctional* nlp = (NonlinearFunctional*)nlProblem;

  FEM::assemble(*nlp->_a, *nlp->_L, *nlp->_A, *nlp->_b, *nlp->_mesh, *nlp->_bc);
  *flag = SAME_NONZERO_PATTERN;
  (*nlp->_A).disp();



  return 0;
}
//-----------------------------------------------------------------------------
