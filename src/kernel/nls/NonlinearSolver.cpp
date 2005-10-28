// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005

#include <petscsnes.h>
#include <petscversion.h>

#include <dolfin/PETScManager.h>
#include <dolfin/NonlinearSolver.h>
#include <dolfin/FEM.h>
#include <dolfin/BilinearForm.h>

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
void NonlinearSolver::solve(Vector& x, NonlinearFunctional& nlfunc)
{
  //FIXME
  // Initiate approximate solution vector
  Vector x0;
  x0.init(x.size());

  x0 = 1;  

  // RHS vector 
  Vector* b;
  b = nlfunc._b;


  Matrix* A;
  A = nlfunc._A;


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
  Due to a bug in PETSc 2.3.0, patch level < 38, we need to check PETSc patch 
  level. If the PETSc version is old, a fatal error is raised when trying to 
  call the PETSc function SNESSetSolution. 
  */
  // Set pointer to approximate solution vector
  #if PETSC_VERSION_PATCH < 38
    dolfin_error("Your version of PETSc is not compatible with the nonlinear "
                 "solver. Download the latest version (2.3.0, patch "
                 "level > 37). Note that the PETSc version numbers are not "
                 "updated. You need a version from after 12 October 2005.");
  #else
    SNESSetSolution(snes, x0.vec());
  #endif

  // Set RHS Function
  SNESSetFunction(snes, b->vec(), FormRHS, &nlfunc);

  // Set Jacobian Function
  SNESSetJacobian(snes, A->mat(), A->mat(), FormJacobian, &nlfunc);

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

  // Pointer to NonlinearFunction object
  NonlinearFunctional* nlp = (NonlinearFunctional*)nlProblem;

//    Mesh mesh  =  *nlp->rmesh();
//   cout << "tetsing here " << (*nlp).rmesh() << endl;;
//   Mesh* AAA;
//   AAA = *nlp._mesh;

  Matrix* AB;
  AB = nlp->_A;

  BilinearForm* a;
  a = nlp->_a;

  // Update functions (user defined)
  nlp->UpdateNonlinearFunction();

  // Assemble RHS vector
//  FEM::assemble(*nlp->_a, *nlp->_L, *nlp->_A, *nlp->_b, *nlp->_mesh, *nlp->_bc);
  FEM::assemble(*a, *nlp->_L, *nlp->_A, *nlp->_b, *nlp->_mesh, *nlp->_bc);
  (*nlp->_A).disp();

//  FEM::assemble(*nlp->_a, *nlp->_L, *nlp->_A, *nlp->_b, *mesh, *nlp->_bc);
  cout << "finish assemble " <<   endl;

  return 0;
}
//-----------------------------------------------------------------------------
int NonlinearSolver::FormJacobian(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlProblem)
{
  cout << "inside FormJacobian " << endl;

  // Pointer to NonlinearFunction object
  NonlinearFunctional* nlp = (NonlinearFunctional*)nlProblem;

  // Assemble Jacobian vector
  FEM::assemble(*nlp->_a, *nlp->_L, *nlp->_A, *nlp->_b, *nlp->_mesh, *nlp->_bc);
  *flag = SAME_NONZERO_PATTERN;



  return 0;
}
//-----------------------------------------------------------------------------
