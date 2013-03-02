// Copyright (C) 2008-2011 Kent-Andre Mardal and Garth N. Wells
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
// Modified by Garth N. Wells 2009
// Modified by Anders Logg 2011-2012
//
// First added:  2008
// Last changed: 2012-08-21

#ifdef HAS_TRILINOS

#include <boost/assign/list_of.hpp>

#include <dolfin/common/MPI.h>

#include <Epetra_ConfigDefs.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_Map.h>
#include <AztecOO.h>
#include <Epetra_LinearProblem.h>

#include <dolfin/log/dolfin_log.h>
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "GenericLinearOperator.h"
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "TrilinosPreconditioner.h"
#include "EpetraKrylovSolver.h"

using namespace dolfin;

// List of available solvers
const std::map<std::string, int> EpetraKrylovSolver::_methods
  = boost::assign::map_list_of("default",  AZ_gmres)
                              ("cg",       AZ_cg)
                              ("gmres",    AZ_gmres)
                              ("tfqmr",    AZ_tfqmr)
                              ("bicgstab", AZ_bicgstab);

// List of available solvers descriptions
const std::vector<std::pair<std::string, std::string> >
EpetraKrylovSolver::_methods_descr = boost::assign::pair_list_of
    ("default",    "default Krylov method")
    ("cg",         "Conjugate gradient method")
    ("gmres",      "Generalized minimal residual method")
    ("tfqmr",      "Transpose-free quasi-minimal residual method")
    ("bicgstab",   "Biconjugate gradient stabilized method");
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EpetraKrylovSolver::methods()
{
  return EpetraKrylovSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EpetraKrylovSolver::preconditioners()
{
  return TrilinosPreconditioner::preconditioners();
}
//-----------------------------------------------------------------------------
Parameters EpetraKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("epetra_krylov_solver");
  p.add("monitor_interval", 1);
  return p;
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(std::string method,
                                       std::string preconditioner)
  : method(method), preconditioner(new TrilinosPreconditioner(preconditioner)),
    solver(new AztecOO), relative_residual(0.0),
    absolute_residual(0.0)

{
  parameters = default_parameters();

  // Check that requsted solver is supported
  if (_methods.count(method) == 0)
  {
    dolfin_error("EpetraKrylovSolver.cpp",
                 "create Epetra Krylov solver",
                 "Unknown Krylov method \"%s\"", method.c_str());
  }

  // Set solver type
  solver->SetAztecOption(AZ_solver, _methods.find(method)->second);
  solver->SetAztecOption(AZ_kspace, parameters("gmres")["restart"]);
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(std::string method,
                                       TrilinosPreconditioner& preconditioner)
  : method(method),  preconditioner(reference_to_no_delete_pointer(preconditioner)),
    solver(new AztecOO), relative_residual(0.0),
    absolute_residual(0.0)
{
  // Set parameter values
  parameters = default_parameters();

  // Check that requsted solver is supported
  if (_methods.count(method) == 0)
  {
    dolfin_error("EpetraKrylovSolver.cpp",
                 "create Epetra Krylov solver",
                 "Unknown Krylov method \"%s\"", method.c_str());
  }

  // Set solver type
  solver->SetAztecOption(AZ_solver, _methods.find(method)->second);
  solver->SetAztecOption(AZ_kspace, parameters("gmres")["restart"]);
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::~EpetraKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EpetraKrylovSolver::set_operator(const boost::shared_ptr<const GenericLinearOperator> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void EpetraKrylovSolver::set_operators(const boost::shared_ptr<const GenericLinearOperator> A,
                                       const boost::shared_ptr<const GenericLinearOperator> P)
{
  this->A = as_type<const EpetraMatrix>(require_matrix(A));
  this->P = as_type<const EpetraMatrix>(require_matrix(P));
  dolfin_assert(this->A);
  dolfin_assert(this->P);
}
//-----------------------------------------------------------------------------
const GenericLinearOperator& EpetraKrylovSolver::get_operator() const
{
  if (!A)
  {
    dolfin_error("EpetraKrylovSolver.cpp",
                 "access operator for Epetra Krylov solver",
                 "Operator has not been set");
  }
  return *A;
}
//-----------------------------------------------------------------------------
std::size_t EpetraKrylovSolver::solve(GenericVector& x,
                                       const GenericVector& b)
{
  return solve(as_type<EpetraVector>(x), as_type<const EpetraVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t EpetraKrylovSolver::solve(EpetraVector& x, const EpetraVector& b)
{
  dolfin_assert(solver);
  dolfin_assert(A);
  dolfin_assert(P);

  // Check dimensions
  const std::size_t M = A->size(0);
  const std::size_t N = A->size(1);
  if (N != b.size())
  {
    dolfin_error("EpetraKrylovSolver.cpp",
                 "solve linear system using Epetra Krylov solver",
                 "Non-matching dimensions for linear system");
  }

  // Write a message
  const bool report = parameters["report"];
  if (report && dolfin::MPI::process_number() == 0)
    info("Solving linear system of size %d x %d (Epetra Krylov solver).", M, N);

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    A->resize(x, 1);
    x.zero();
  }
  else if (!parameters["nonzero_initial_guess"])
    x.zero();

  // Create linear problem
  Epetra_LinearProblem linear_problem(A->mat().get(), x.vec().get(),
                                      b.vec().get());
  // Set-up linear solver
  solver->SetProblem(linear_problem);

  // Set output level
  if (parameters["monitor_convergence"])
  {
    const std::size_t interval = parameters["monitor_interval"];
    solver->SetAztecOption(AZ_output, interval);
  }
  else
    solver->SetAztecOption(AZ_output, AZ_none);

  dolfin_assert(P);
  preconditioner->set(*this, *P);

  // Set covergence check method
  solver->SetAztecOption(AZ_conv, AZ_rhs);

  // Start solve
  solver->Iterate(static_cast<int>(parameters["maximum_iterations"]),
                  parameters["relative_tolerance"]);

  // Check solve status
  const double* status = solver->GetAztecStatus();
  if ((int) status[AZ_why] != AZ_normal)
  {
    std::string errorDescription;
    if( status[AZ_why] == AZ_maxits )
      errorDescription = "maximum iters reached";
    else if( status[AZ_why] == AZ_loss )
      errorDescription = "loss of accuracy";
    else if( status[AZ_why] == AZ_ill_cond  )
      errorDescription = "ill-conditioned";
    else if( status[AZ_why] == AZ_breakdown )
      errorDescription = "breakdown";
    else
      errorDescription = "unknown error";

    std::stringstream message;
    message << "Epetra (AztecOO) Krylov solver (" << method << ", " << preconditioner->name() << ") "
            << "failed to converge after " << (int)status[AZ_its] << " iterations "
            << "(" << errorDescription << ", error code " << (int)status[AZ_why] << ")";

    if (parameters["error_on_nonconvergence"])
    {
      dolfin_error("EpetraKrylovSolver.cpp",
                   "solve linear system using Epetra Krylov solver",
                   message.str());
    }
    else
      warning(message.str());
  }
  else
  {
    info("Epetra (AztecOO) Krylov solver (%s, %s) converged in %d iterations.",
          method.c_str(), preconditioner->name().c_str(), solver->NumIters());
  }

  // Update residuals
  absolute_residual = solver->TrueResidual();
  relative_residual = solver->ScaledResidual();

  // Return number of iterations
  return solver->NumIters();
}
//-----------------------------------------------------------------------------
std::size_t EpetraKrylovSolver::solve(const GenericLinearOperator& A,
                                       GenericVector& x,
                                       const GenericVector& b)
{
  return solve(as_type<const EpetraMatrix>(require_matrix(A)),
               as_type<EpetraVector>(x),
               as_type<const EpetraVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t EpetraKrylovSolver::solve(const EpetraMatrix& A, EpetraVector& x,
                                       const EpetraVector& b)
{
  boost::shared_ptr<const EpetraMatrix> _A(&A, NoDeleter());
  set_operator(_A);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
double EpetraKrylovSolver::residual(const std::string residual_type) const
{
  if (residual_type == "relative")
    return relative_residual;
  else if (residual_type == "absolute")
    return absolute_residual;
  else
  {
    dolfin_error("EpetraKrylovSolver.cpp",
                 "compute residual of Epetra Krylov solver",
                 "Unknown residual type: \"%s\"", residual_type.c_str());
    return 0.0;
  }
}
//-----------------------------------------------------------------------------
std::string EpetraKrylovSolver::str(bool verbose) const
{
  dolfin_not_implemented();
  return std::string();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<AztecOO> EpetraKrylovSolver::aztecoo() const
{
  return solver;
}
//-----------------------------------------------------------------------------
#endif
