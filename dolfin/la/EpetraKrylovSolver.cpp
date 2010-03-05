// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// Last changed: 2009-09-08

#ifdef HAS_TRILINOS

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

#include <boost/assign/list_of.hpp>
#include <dolfin/log/dolfin_log.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "EpetraKrylovSolver.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "TrilinosPreconditioner.h"
#include "KrylovSolver.h"

using namespace dolfin;

// Available solvers
const std::map<std::string, int> EpetraKrylovSolver::methods
  = boost::assign::map_list_of("default",  AZ_gmres)
                              ("cg",       AZ_cg)
                              ("gmres",    AZ_gmres)
                              ("bicgstab", AZ_bicgstab);
//-----------------------------------------------------------------------------
Parameters EpetraKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("epetra_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(std::string method, std::string pc_type)
                    : method(method),
                      preconditioner(new TrilinosPreconditioner(pc_type)),
                      solver(new AztecOO)
{
  parameters = default_parameters();

  // Check that requsted solver is supported
  if (methods.count(method) == 0 )
    error("Requested EpetraKrylovSolver method '%s' in unknown", method.c_str());
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(std::string method,
                  TrilinosPreconditioner& preconditioner)
                : method(method),
                  preconditioner(reference_to_no_delete_pointer(preconditioner)),
                  solver(new AztecOO)
{
  // Set parameter values
  parameters = default_parameters();

  // Check that requsted solver is supported
  if (methods.count(method) == 0 )
    error("Requested EpetraKrylovSolver method '%s' in unknown", method.c_str());
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::~EpetraKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                       const GenericVector& b)
{
  return solve(A.down_cast<EpetraMatrix>(), x.down_cast<EpetraVector>(),
               b.down_cast<EpetraVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraKrylovSolver::solve(const EpetraMatrix& A, EpetraVector& x,
                                       const EpetraVector& b)
{
  assert(solver);

  // FIXME: This function needs to be cleaned up

  // Check dimensions
  const uint M = A.size(0);
  const uint N = A.size(1);
  if (N != b.size())
    error("Non-matching dimensions for linear system.");

  // Write a message
  if (parameters["report"])
    info("Solving linear system of size %d x %d (Epetra Krylov solver).", M, N);

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    x.resize(M);
    x.zero();
  }

  // FIXME: permit initial guess

  // Create linear problem
  Epetra_LinearProblem linear_problem(A.mat().get(), x.vec().get(),
                                      b.vec().get());
  // Set-up linear solver
  solver->SetProblem(linear_problem);

  // Set solver type
  solver->SetAztecOption(AZ_solver, methods.find(method)->second);
  solver->SetAztecOption(AZ_kspace, parameters["gmres_restart"]);

  // Set output level
  if(parameters["monitor_convergence"])
    solver->SetAztecOption(AZ_output, 1);
  else
    solver->SetAztecOption(AZ_output, AZ_none);

  // Configure preconditioner
  preconditioner->set(*this);

  // Start solve
  solver->Iterate(parameters["maximum_iterations"], parameters["relative_tolerance"]);
  const double* status = solver->GetAztecStatus();
  if ( (int) status[AZ_why] != AZ_normal )
    warning("Problem with Trilinos Krylov solver. Error code %i.", status[AZ_why]);
  else
  {
    info("AztecOO Krylov solver (%s, %s) converged in %d iterations.",
          method.c_str(), preconditioner->name().c_str(), solver->NumIters());
  }

  return solver->NumIters();
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
