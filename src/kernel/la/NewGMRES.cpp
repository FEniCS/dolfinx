// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <petsc/petscksp.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/VirtualMatrix.h>
#include <dolfin/NewGMRES.h>

#include <petsc/petscpc.h>
#include <dolfin/pcimpl.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewGMRES::NewGMRES()
{
  // Initialize PETSc
  PETScManager::init();

  //Set up solver environment.
  KSPCreate(PETSC_COMM_WORLD, &ksp);

  dolfin::cout << "Setting up PETSc solver environment." << dolfin::endl;
}
//-----------------------------------------------------------------------------
NewGMRES::~NewGMRES()
{
  //Destroy solver environment.
  KSPDestroy(ksp);
}
//-----------------------------------------------------------------------------
void NewGMRES::solve(const NewMatrix& A, NewVector& x, const NewVector& b)
{
  //KSP  ksp;
  //PC   pc;

  // The default ILU preconditioner creates an extra matrix.
  // To save memory, use e.g. a Jacobi preconditioner.

  KSPSetOperators(ksp,A.mat(),A.mat(),DIFFERENT_NONZERO_PATTERN);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  //KSPSetFromOptions(ksp);

  /*
  // Set tolerances
  real rtol,abstol,dtol;
  int maxits; 
  KSPGetTolerances(ksp,&rtol,&abstol,&dtol,&maxits);
  KSPSetTolerances(ksp,rtol,abstol,dtol,maxits);
  */

  // Solve system
  dolfin::cout << "Solving system using KSPSolve()." << dolfin::endl;

  //PC pc;
  //KSPGetPC(ksp, &pc);
  //PCSetType(pc, PCNONE);

  KSPSetRhs(ksp, b.vec());
  KSPSetSolution(ksp, x.vec());

  KSPSolve(ksp);

  int its = 0;
  KSPGetIterationNumber(ksp, &its);
  dolfin_info("GMRES converged in %d iterations.", its);

  KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
}
//-----------------------------------------------------------------------------
void NewGMRES::solve(const VirtualMatrix& A, NewVector& x, const NewVector& b)
{
  // The default ILU preconditioner creates an extra matrix.
  // To save memory, use e.g. a Jacobi preconditioner.
  
  KSPSetOperators(ksp,A.mat(),A.mat(),DIFFERENT_NONZERO_PATTERN);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  KSPSetFromOptions(ksp);

  /*
  // Set tolerances
  real rtol,abstol,dtol;
  int maxits; 
  KSPGetTolerances(ksp,&rtol,&abstol,&dtol,&maxits);
  KSPSetTolerances(ksp,rtol,abstol,dtol,maxits);
  */

  //Solve system.
  dolfin::cout << "Solving system using KSPSolve()." << dolfin::endl;

  KSPSetRhs(ksp, b.vec());
  KSPSetSolution(ksp, x.vec());
  KSPSolve(ksp);

  int its = 0;
  KSPGetIterationNumber(ksp, &its);
  dolfin_info("GMRES converged in %d iterations.", its);

  //KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
}
//-----------------------------------------------------------------------------
void NewGMRES::setRtol(real rt)
{
  KSPGetTolerances(ksp,&rtol,&abstol,&dtol,&maxits);
  rtol = rt;
  KSPSetTolerances(ksp,rtol,abstol,dtol,maxits);
}
//-----------------------------------------------------------------------------
void NewGMRES::setAbstol(real at)
{
  KSPGetTolerances(ksp,&rtol,&abstol,&dtol,&maxits);
  abstol = at;
  KSPSetTolerances(ksp,rtol,abstol,dtol,maxits);
}
//-----------------------------------------------------------------------------
void NewGMRES::setDtol(real dt)
{
  KSPGetTolerances(ksp,&rtol,&abstol,&dtol,&maxits);
  dtol = dt;
  KSPSetTolerances(ksp,rtol,abstol,dtol,maxits);
}
//-----------------------------------------------------------------------------
void NewGMRES::setMaxits(int mi)
{
  KSPGetTolerances(ksp,&rtol,&abstol,&dtol,&maxits);
  maxits = mi;
  KSPSetTolerances(ksp,rtol,abstol,dtol,maxits);
}
//-----------------------------------------------------------------------------
void NewGMRES::setPreconditioner(NewPreconditioner &pc)
{
  PC petscpc;
  KSPGetPC(ksp, &petscpc);

  NewPreconditioner::PCCreate(petscpc);

  petscpc->data = &pc;
  petscpc->ops->apply = NewPreconditioner::PCApply;
}
//-----------------------------------------------------------------------------
