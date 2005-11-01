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
dolfin::uint NonlinearSolver::solve(Vector& x, NonlinearFunction& nonlinear_function)
{

  Matrix A;
  Vector b, r;

  BilinearForm& a       = nonlinear_function.a();
  BoundaryCondition& bc = nonlinear_function.bc();
  Mesh& mesh = nonlinear_function.mesh();

  // Initialize global RHS vector, Jacobian matrix, solution vector and approximate 
  // solution vector
  const uint M = FEM::size(mesh, a.test());
  const uint N = FEM::size(mesh, a.trial());
  const uint nz = FEM::nzsize(mesh, a.trial());
  A.init(M, N, nz);
  b.init(M);
  x.init(M);
  r.init(M);

  FEM::applyBC(A, r, mesh, a.trial(), bc);

  // Store address of Jacobian matrix so that it can be assembled at same time as F(u)
  nonlinear_function._A = &A;
  nonlinear_function._b = &b;

  // Set constant r (F(u) = r) (used for setting inhomeneous Dirichlet BC
  SNESSetRhs(snes, r.vec());

  // Set nonlinar function and Jacobian function (two functions)
//  SNESSetFunction(snes, b.vec(), formSystem, &nonlinear_function);
//  SNESSetJacobian(snes, A.mat(), A.mat(), formDummy, &nonlinear_function);


///  FIXME - implement separate functions for computing RHS and Jacobian. 
///  Application of boundary conditions requires work.

  // Set nonlinar function and Jacobian function (in one function)
  SNESSetFunction(snes, b.vec(), formRHS, &nonlinear_function);
  SNESSetJacobian(snes, A.mat(), A.mat(), formJacobian, &nonlinear_function);


  // Set options for linear solver
  KSP ksp;   
  SNESGetKSP(snes, &ksp);
  KSPSetTolerances(ksp, 1.0e-15, PETSC_DEFAULT, PETSC_DEFAULT, 10000);

  // Set preconditioner type
  PC  pc; 
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCILU);

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

  // Objects within NonlinearFunction
  BilinearForm& a = nonlinear_function->a();
  LinearForm& L   = nonlinear_function->L();
  BoundaryCondition& bc = nonlinear_function->bc();
  Mesh& mesh = nonlinear_function->mesh();

  Matrix& A  = *(nonlinear_function->_A);

  Vector b(f);

  // Update nonlinear function (user defined)
  Vector xsol(x);
  nonlinear_function->update(xsol);
  
  //FIXME - should assmble vector b only
  // Assemble RHS vector
  FEM::assemble(a, L, A, b, mesh, bc);

  return 0;
}
//-----------------------------------------------------------------------------
int NonlinearSolver::formJacobian(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlProblem)
{
  // Pointer to NonlinearFunction
  NonlinearFunction* nonlinear_function = (NonlinearFunction*)nlProblem;

  // Set pointers to objects within NonlinearFunction
  // Objects within NonlinearFunction
  BilinearForm& a = nonlinear_function->a();
  LinearForm& L   = nonlinear_function->L();
  BoundaryCondition& bc = nonlinear_function->bc();
  Mesh& mesh = nonlinear_function->mesh();

  Matrix& A  = *(nonlinear_function->_A);
  Vector& b  = *(nonlinear_function->_b);

  //FIXME - should assmble matrix A only
  // Assemble Jacobian vector
  FEM::assemble(a, L, A, b, mesh, bc);

  // Structure of returned matrix
  *flag = SAME_NONZERO_PATTERN;

  return 0;
}
//-----------------------------------------------------------------------------
int NonlinearSolver::formSystem(SNES snes, Vec x, Vec f, void *nlProblem)
{
  // Pointer to nonlinear function
  NonlinearFunction* nonlinear_function = (NonlinearFunction*)nlProblem;

  // Objects within NonlinearFunction
  BilinearForm& a = nonlinear_function->a();
  LinearForm& L   = nonlinear_function->L();
  BoundaryCondition& bc = nonlinear_function->bc();
  Mesh& mesh = nonlinear_function->mesh();

  Matrix& A  = *(nonlinear_function->_A);

  // Vector for F(u)
  Vector b(f);

  // Update nonlinear function (user defined)
  Vector xsol(x);
  nonlinear_function->update(xsol);
  
  // Assemble RHS vector, Jacobian, and apply boundary conditions 
  FEM::assemble(a, L, A, b, mesh, bc);

  return 0;
}
//-----------------------------------------------------------------------------
int NonlinearSolver::formDummy(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlProblem)
{
  // Dummy function for computing Jacobian. Do nothing.
  return 0;
}
//-----------------------------------------------------------------------------
