// Copyright (C) 2008-2010 Kent-Andre Mardal and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2010-02-21

#ifdef HAS_TRILINOS

#include <boost/scoped_ptr.hpp>

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
EpetraLUSolver::EpetraLUSolver()
{
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
EpetraLUSolver::~EpetraLUSolver()
{
  // Do nothing
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
  // Create linear problem
  Epetra_LinearProblem linear_problem(A.mat().get(), x.vec().get(),
                                      b.vec().get());

  // Create linear solver
  Amesos factory;
  std::string solver_type;
  solver_type = "Amesos_Superludist";
  /*
  if (factory.Query("Amesos_Mumps"))
  {
    cout <<  "Using MUMPS" << endl;
    solver_type = "Amesos_Mumps";
  }
  */
  /*
  if (factory.Query("Amesos_Umfpack"))
    solver_type = "Amesos_Umfpack";
  else if (factory.Query("Amesos_Klu"))
    solver_type = "Amesos_Klu";
  else
    error("Requested LU solver not available");
  */
  boost::scoped_ptr<Amesos_BaseSolver> solver(factory.Create(solver_type, linear_problem));

  // Factorise matrix
  AMESOS_CHK_ERR(solver->SymbolicFactorization());
  AMESOS_CHK_ERR(solver->NumericFactorization());

  // Solve
  AMESOS_CHK_ERR(solver->Solve());

  return 1;
}
//-----------------------------------------------------------------------------
std::string EpetraLUSolver::str(bool verbose) const
{
  dolfin_not_implemented();
  return std::string();
}
//-----------------------------------------------------------------------------
#endif
