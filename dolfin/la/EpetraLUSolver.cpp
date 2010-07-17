// Copyright (C) 2008-2010 Kent-Andre Mardal and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2010-02-21

#ifdef HAS_TRILINOS

#include <Amesos.h>
#include <Amesos_BaseSolver.h>
#include <Amesos_ConfigDefs.h>
#include <Amesos_Klu.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_MultiVector.h>

#include "GenericMatrix.h"
#include "GenericVector.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "LUSolver.h"
#include "EpetraLUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters EpetraLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("epetra_lu_solver");
  return p;
}
//-----------------------------------------------------------------------------
EpetraLUSolver::EpetraLUSolver() : linear_problem(new Epetra_LinearProblem)
{
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
EpetraLUSolver::EpetraLUSolver(const GenericMatrix& A)
                             : linear_problem(new Epetra_LinearProblem)
{
  parameters = default_parameters();
  set_operator(A);
}
//-----------------------------------------------------------------------------
EpetraLUSolver::~EpetraLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EpetraLUSolver::set_operator(const GenericMatrix& A)
{
  assert(linear_problem);
  const EpetraMatrix& _A = A.down_cast<EpetraMatrix>();
  linear_problem->SetOperator(_A.mat().get());
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(linear_problem);

  // Downcast vector
  EpetraVector& _x = x.down_cast<EpetraVector>();
  const EpetraVector& _b = b.down_cast<EpetraVector>();

  // Get operator matrix
  const Epetra_RowMatrix* A =	linear_problem->GetMatrix();
  if (!A)
    error("Operator has not been set for EpetraLUSolver.");

  const uint M = A->NumGlobalRows();
  const uint N = A->NumGlobalCols();
  if (N != b.size())
    error("Non-matching dimensions for linear system.");

  // Initialize solution vector (remains untouched if dimensions match)
  x.resize(M);

  // Set LHS and RHS vectors
  linear_problem->SetRHS(_b.vec().get());
  linear_problem->SetLHS(_x.vec().get());

  // Create linear solver
  Amesos factory;
  std::string solver_type;
  if (factory.Query("Amesos_Mumps"))
    solver_type = "Amesos_Mumps";
  else if (factory.Query("Amesos_Umfpack"))
    solver_type = "Amesos_Umfpack";
  else if (factory.Query("Amesos_Klu"))
    solver_type = "Amesos_Klu";
  else
    error("Requested LU solver not available");
  boost::scoped_ptr<Amesos_BaseSolver> solver(factory.Create(solver_type, *linear_problem));

  // Factorise matrix
  AMESOS_CHK_ERR(solver->SymbolicFactorization());
  AMESOS_CHK_ERR(solver->NumericFactorization());

  // Solve
  AMESOS_CHK_ERR(solver->Solve());

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(const GenericMatrix& A, GenericVector& x,
                                   const GenericVector& b)
{
  return solve(A.down_cast<EpetraMatrix>(), x.down_cast<EpetraVector>(),
               b.down_cast<EpetraVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(const EpetraMatrix& A, EpetraVector& x,
                                   const EpetraVector& b)
{
  set_operator(A);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
std::string EpetraLUSolver::str(bool verbose) const
{
  dolfin_not_implemented();
  return std::string();
}
//-----------------------------------------------------------------------------
#endif
