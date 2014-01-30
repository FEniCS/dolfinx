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

#include <Epetra_Comm.h>
#include <Epetra_ConfigDefs.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_Map.h>
#include <Epetra_LinearProblem.h>
#include <BelosLinearProblem.hpp>
#include <BelosEpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

#include <dolfin/common/Timer.h>
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
const std::map<std::string, std::string> EpetraKrylovSolver::_methods
  = boost::assign::map_list_of("default", "GMRES")
                              ("cg",      "CG")
                              ("gmres",   "GMRES")
                              ("minres",  "MINRES")
                              ("rcg",     "RCG")
                              ("gcrodr",  "GCRODR")
                              ("lsqr",    "LSQR");

// List of available solvers descriptions
const std::vector<std::pair<std::string, std::string> >
EpetraKrylovSolver::_methods_descr = boost::assign::pair_list_of
    ("default", "default Krylov method")
    ("cg",      "Conjugate gradient method")
    ("gmres",   "Generalized minimal residual method")
    ("minres",  "Minimal residual method")
    ("rcg",     "Recycling conjugate gradient method")
    ("gcrodr",  "Block recycling GMRES method")
    ("lsqr",    "Least-squared QR method")
    ;
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
  : _method(method),
    _preconditioner(new TrilinosPreconditioner(preconditioner)),
    _relative_residual(0.0),
    _absolute_residual(0.0)
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
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(std::string method,
                                       TrilinosPreconditioner& preconditioner)
  : _method(method),
    _preconditioner(reference_to_no_delete_pointer(preconditioner)),
     _relative_residual(0.0), _absolute_residual(0.0)
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
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::~EpetraKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void EpetraKrylovSolver::set_operator(
    const boost::shared_ptr<const GenericLinearOperator> A
    )
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void EpetraKrylovSolver::set_operators(
    const boost::shared_ptr<const GenericLinearOperator> A,
    const boost::shared_ptr<const GenericLinearOperator> P
    )
{
  _A = as_type<const EpetraMatrix>(require_matrix(A));
  _P = as_type<const EpetraMatrix>(require_matrix(P));
  dolfin_assert(_A);
  dolfin_assert(_P);
}
//-----------------------------------------------------------------------------
const GenericLinearOperator& EpetraKrylovSolver::get_operator() const
{
  if (!_A)
  {
    dolfin_error("EpetraKrylovSolver.cpp",
                 "access operator for Epetra Krylov solver",
                 "Operator has not been set");
  }
  return *_A;
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
  Timer timer("Epetra Krylov solver");

  dolfin_assert(solver);
  dolfin_assert(_A);
  dolfin_assert(_P);

  // Check dimensions
  const std::size_t M = _A->size(0);
  const std::size_t N = _A->size(1);
  if (N != b.size())
  {
    dolfin_error("EpetraKrylovSolver.cpp",
                 "solve linear system using Epetra Krylov solver",
                 "Non-matching dimensions for linear system");
  }

  // Write a message
  const bool report = parameters["report"];
  if (report && _A->mat()->Comm().MyPID() == 0)
    info("Solving linear system of size %d x %d (Epetra Krylov solver).",
         M, N
         );

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    _A->resize(x, 1);
    x.zero();
  }
  else if (!parameters["nonzero_initial_guess"])
    x.zero();

  // Create linear problem
  // Make clear that the RCP doesn't own the memory and thus doesn't try
  // to destroy the object when it goes out of scope.
  Teuchos::RCP<Belos::LinearProblem<ST,MV,OP> > linear_problem =
    Teuchos::rcp(new Belos::LinearProblem<ST,MV,OP>(
          Teuchos::rcp(_A->mat().get(), false),
          Teuchos::rcp(x.vec().get(), false),
          Teuchos::rcp(b.vec().get(), false)
          ));
  const int ierr = linear_problem->setProblem();
  dolfin_assert(ierr == 0);

  Teuchos::ParameterList belosList;
  // Set output level
  if (parameters["monitor_convergence"])
  {
    belosList.set("Verbosity", Belos::Errors + Belos::Warnings);
    belosList.set("Output Frequency", (int)parameters["monitor_interval"]);
  }

  belosList.set("Convergence Tolerance",
                (double)parameters["relative_tolerance"]
                );
  belosList.set("Maximum Iterations", (int)parameters["maximum_iterations"]);

  // Set preconditioner
  dolfin_assert(_P);
  _preconditioner->set(*linear_problem, *_P);

  // Look up the Belos name of the method in _methods. This is a little
  // complicated since std::maps<> don't have const lookup.
  std::map<std::string, std::string>::const_iterator it =
    _methods.find(_method);
  if (it == _methods.end())
      dolfin_error("EpetraKrylovSolver.cpp",
                   "solve linear system using Epetra Krylov solver",
                   "unknown method \"%s\"", _method.c_str());
  // Set-up linear solver
  Belos::SolverFactory<ST,MV,OP> factory;
  Teuchos::RCP<Belos::SolverManager<ST,MV,OP> > solver =
    factory.create(it->second, Teuchos::rcp(&belosList, false));
  solver->setProblem(linear_problem);

  // Start solve
  Belos::ReturnType ret = solver->solve();

  if (ret == Belos::Converged)
  {
    info("Epetra (Belos) Krylov solver (%s, %s) converged in %d iterations.",
         _method.c_str(),
         _preconditioner->name().c_str(),
         solver->getNumIters()
         );
  }
  else if (ret == Belos::Unconverged)
  {
    std::stringstream message;
    message << "Epetra (Belos) Krylov solver (" << _method << ", "
            << _preconditioner->name() << ") "
            << "failed to converge after " << solver->getNumIters()
            << " iterations ";
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
    dolfin_error("EpetraKrylovSolver.cpp",
                 "solve linear system using Epetra Krylov solver",
                 "illegal Belos return code");
  }

  // TODO
  // Update residuals
  //_absolute_residual = solver->TrueResidual();
  //_relative_residual = solver->ScaledResidual();

  // Return number of iterations
  return solver->getNumIters();
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
  boost::shared_ptr<const EpetraMatrix> Atmp(&A, NoDeleter());
  set_operator(Atmp);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
double EpetraKrylovSolver::residual(const std::string residual_type) const
{
  dolfin_error("EpetraKrylovSolver.cpp",
               "compute residual of Epetra Krylov solver",
               "Not yet implemented");

  if (residual_type == "relative")
    return _relative_residual;
  else if (residual_type == "absolute")
    return _absolute_residual;
  else
  {
    dolfin_error("EpetraKrylovSolver.cpp",
                 "compute residual of Epetra Krylov solver",
                 "Unknown residual type: \"%s\"", residual_type.c_str()
                 );
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
#endif
