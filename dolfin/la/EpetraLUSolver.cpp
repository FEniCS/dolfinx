// Copyright (C) 2008-2010 Kent-Andre Mardal and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2011.
//
// Last changed: 2011-10-07

#ifdef HAS_TRILINOS

#include <Amesos.h>
#include <Amesos_BaseSolver.h>
#include <Amesos_ConfigDefs.h>
#include <Amesos_Klu.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_MultiVector.h>

#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "LUSolver.h"
#include "EpetraLUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EpetraLUSolver::list_methods()
{
  static std::vector<std::pair<std::string, std::string> > m;

  m.push_back(std::make_pair("default",      "default LU solver"));
  m.push_back(std::make_pair("umfpack",      "UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)"));
  m.push_back(std::make_pair("mumps",        "MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)"));
  m.push_back(std::make_pair("klu",          "Trilinos KLU"));

  return m;
}
//-----------------------------------------------------------------------------
std::string EpetraLUSolver::choose_method(std::string method) const
{
  Amesos factory;
  if (method == "default")
  {
    if (factory.Query("Amesos_Mumps"))
      method = "Amesos_Mumps";
    else if (factory.Query("Amesos_Umfpack"))
      method = "Amesos_Umfpack";
    else if (factory.Query("Amesos_Klu"))
      method = "Amesos_Klu";
  }
  else if (method == "umfpack")
    method = "Amesos_Umfpack";
  else if (method == "mumps")
    method = "Amesos_mumps";
  else if (method == "klu")
    method = "Amesos_klu";
  else
  {
    dolfin_error("EpetraLUSolver.cpp",
                 "solve linear system",
                 "Unknown LU solver method \"%s\". "
                 "Use list_lu_methods() to list available methods",
                 method.c_str());
  }

  return method;
}
//-----------------------------------------------------------------------------
Parameters EpetraLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("epetra_lu_solver");
  return p;
}
//-----------------------------------------------------------------------------
EpetraLUSolver::EpetraLUSolver(std::string method)
  : symbolic_factorized(false),
    numeric_factorized(false),
    linear_problem(new Epetra_LinearProblem)
{
  // Set default parameters
  parameters = default_parameters();

  // Choose method
  method = choose_method(method);

  // Initialize solver
  Amesos factory;
  solver.reset(factory.Create(method, *linear_problem));
}
//-----------------------------------------------------------------------------
EpetraLUSolver::EpetraLUSolver(boost::shared_ptr<const GenericMatrix> A,
                               std::string method)
  : symbolic_factorized(false),
    numeric_factorized(false),
    linear_problem(new Epetra_LinearProblem)
{
  // Set default parameters
  parameters = default_parameters();

  // Set operator
  this->A = GenericTensor::down_cast<const EpetraMatrix>(A);
  assert(this->A);

  // Choose method
  method = choose_method(method);

  // Initialize solver
  Amesos factory;
  solver.reset(factory.Create(method, *linear_problem));
}
//-----------------------------------------------------------------------------
EpetraLUSolver::~EpetraLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EpetraLUSolver::set_operator(const boost::shared_ptr<const GenericMatrix> A)
{
  assert(linear_problem);

  this->A = GenericTensor::down_cast<const EpetraMatrix>(A);
  assert(this->A);
  linear_problem->SetOperator(this->A->mat().get());

  symbolic_factorized = false;
  numeric_factorized  = false;
}
//-----------------------------------------------------------------------------
const GenericMatrix& EpetraLUSolver::get_operator() const
{
  if (!A)
    error("Operator for linear solver has not been set.");
  return *A;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(linear_problem);
  assert(solver);

  // Write a message
  if (parameters["report"] && dolfin::MPI::process_number() == 0)
    log(PROGRESS, "Solving linear system of size %d x %d (Epetra LU solver).", A->size(0), A->size(1));

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

  // Initialize solution vector
  if (x.size() != M)
  {
    this->A->resize(x, 1);
    x.zero();
  }

  // Set LHS and RHS vectors
  linear_problem->SetRHS(_b.vec().get());
  linear_problem->SetLHS(_x.vec().get());

  // Get some parameters
  const bool reuse_fact   = parameters["reuse_factorization"];
  const bool same_pattern = parameters["same_nonzero_pattern"];

  // Perform symbolic factorization
  if ( (reuse_fact || same_pattern) && !symbolic_factorized )
  {
    AMESOS_CHK_ERR(solver->SymbolicFactorization());
    symbolic_factorized = true;
  }
  else if (!reuse_fact && !same_pattern)
  {
    AMESOS_CHK_ERR(solver->SymbolicFactorization());
    symbolic_factorized = true;
  }

  // Perform numeric factorization
  if (reuse_fact && !numeric_factorized)
  {
    AMESOS_CHK_ERR(solver->NumericFactorization());
    numeric_factorized = true;
  }
  else if (!reuse_fact)
  {
    AMESOS_CHK_ERR(solver->NumericFactorization());
    numeric_factorized = true;
  }

  log(PROGRESS, "Solving linear system of size %d x %d (Trilinos LU solver (%s)).",
      A->NumGlobalRows(), A->NumGlobalCols(), method.c_str());

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
  boost::shared_ptr<const EpetraMatrix> _A(&A, NoDeleter());
  set_operator(_A);
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
