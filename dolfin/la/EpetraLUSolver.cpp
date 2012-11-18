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
// Modified by Anders Logg 2011-2012
//
// First added:  2008
// Last changed: 2012-08-22

#ifdef HAS_TRILINOS

// Included here to avoid a C++ problem with some MPI implementations
#include <dolfin/common/MPI.h>

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
#include "GenericLinearOperator.h"
#include "GenericVector.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "LUSolver.h"
#include "EpetraLUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EpetraLUSolver::methods()
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
    else
    {
      dolfin_error("EpetraLUSolver.cpp",
                   "choose default Epetra LU solver",
                   "No methods available");
    }
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
                 "Use list_lu_solver_methods() to list available methods",
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
  this->method = choose_method(method);

  // Initialize solver
  Amesos factory;
  solver.reset(factory.Create(this->method, *linear_problem));

  // Check that solver was initialized correctly
  if (!solver)
  {
    dolfin_error("EpetraLUSolver.cpp",
                 "create Epetra LU solver",
                 "Epetra was not able to create linear solver \"%s\"",
                 this->method.c_str());
  }
}
//-----------------------------------------------------------------------------
EpetraLUSolver::EpetraLUSolver(boost::shared_ptr<const GenericLinearOperator> A,
                               std::string method)
  : symbolic_factorized(false),
    numeric_factorized(false),
    linear_problem(new Epetra_LinearProblem)
{
  // Set default parameters
  parameters = default_parameters();

  // Set operator
  this->A = as_type<const EpetraMatrix>(require_matrix(A));
  dolfin_assert(this->A);

  // Choose method
  this->method = choose_method(method);

  // Initialize solver
  Amesos factory;
  solver.reset(factory.Create(this->method, *linear_problem));

  // Check that solver was initialized correctly
  if (!solver)
  {
    dolfin_error("EpetraLUSolver.cpp",
                 "create Epetra LU solver",
                 "Epetra was not able to create linear solver \"%s\"",
                 this->method.c_str());
  }
}
//-----------------------------------------------------------------------------
EpetraLUSolver::~EpetraLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EpetraLUSolver::set_operator(const boost::shared_ptr<const GenericLinearOperator> A)
{
  dolfin_assert(linear_problem);

  this->A = as_type<const EpetraMatrix>(require_matrix(A));
  dolfin_assert(this->A);
  linear_problem->SetOperator(this->A->mat().get());

  symbolic_factorized = false;
  numeric_factorized  = false;
}
//-----------------------------------------------------------------------------
const GenericLinearOperator& EpetraLUSolver::get_operator() const
{
  if (!A)
  {
    dolfin_error("EpetraLUSolver.cpp",
                 "access operator for Epetra LU solver",
                 "Operator has not been set");
  }
  return *A;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  dolfin_assert(linear_problem);
  dolfin_assert(solver);

  // Write a message
  if (parameters["report"] && dolfin::MPI::process_number() == 0)
  {
    info("Solving linear system of size %d x %d using Epetra LU solver (%s).",
         A->size(0), A->size(1), method.c_str());
  }

  // Downcast vector
  EpetraVector& _x = as_type<EpetraVector>(x);
  const EpetraVector& _b = as_type<const EpetraVector>(b);

  // Get operator matrix
  const Epetra_RowMatrix* A = linear_problem->GetMatrix();
  if (!A)
  {
    dolfin_error("EpetraLUSolver.cpp",
                 "solve linear system using Epetra LU solver",
                 "Operator has not been set");
  }

  const std::size_t M = A->NumGlobalRows64();
  const std::size_t N = A->NumGlobalCols64();
  if (N != b.size())
  {
    dolfin_error("EpetraLUSolver.cpp",
                 "solve linear system using Epetra LU solver",
                 "Non-matching dimensions for linear system");
  }

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

  // Solve
  AMESOS_CHK_ERR(solver->Solve());

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraLUSolver::solve(const GenericLinearOperator& A,
                                   GenericVector& x,
                                   const GenericVector& b)
{
  return solve(as_type<const EpetraMatrix>(require_matrix(A)),
               as_type<EpetraVector>(x),
               as_type<const EpetraVector>(b));
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
