// Copyright (C) 2007-2010 Garth N. Wells
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
// Modified by Ola Skavhaug 2008
// Modified by Anders Logg 2008-2012
//
// First added:  2007-07-03
// Last changed: 2013-11-25

#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/log/LogStream.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "DefaultFactory.h"
#include "LinearSolver.h"
#include "VectorSpaceBasis.h"
#include "KrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters KrylovSolver::default_parameters()
{
  Parameters p("krylov_solver");

  p.add<double>("relative_tolerance");
  p.add<double>("absolute_tolerance");
  p.add<double>("divergence_limit");
  p.add<int>("maximum_iterations");
  p.add<bool>("report");
  p.add<bool>("monitor_convergence");
  p.add<bool>("error_on_nonconvergence");
  p.add<bool>("nonzero_initial_guess");

  return p;
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(MPI_Comm comm, std::string method, std::string preconditioner)
{
  // Initialize solver
  init(method, preconditioner, comm);
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(std::string method, std::string preconditioner)
{
  // Initialize solver
  init(method, preconditioner, MPI_COMM_WORLD);
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(MPI_Comm comm,
                           std::shared_ptr<const GenericLinearOperator> A,
                           std::string method, std::string preconditioner)
{
  // Initialize solver
  init(method, preconditioner, comm);

  // Set operator
  set_operator(A);
}
//-----------------------------------------------------------------------------
KrylovSolver::~KrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void
KrylovSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  dolfin_assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
void
KrylovSolver::set_operators(std::shared_ptr<const GenericLinearOperator> A,
                            std::shared_ptr<const GenericLinearOperator> P)
{
  dolfin_assert(solver);
  solver->parameters.update(parameters);
  solver->set_operators(A, P);
}
//-----------------------------------------------------------------------------
std::size_t KrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  dolfin_assert(solver);
  //check_dimensions(solver->get_operator(), x, b);

  Timer timer("Krylov solver");
  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
std::size_t KrylovSolver::solve(const GenericLinearOperator& A,
                                GenericVector& x,
                                const GenericVector& b)
{
  dolfin_assert(solver);
  //check_dimensions(A, x, b);

  Timer timer("Krylov solver");
  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
void KrylovSolver::init(std::string method, std::string preconditioner, MPI_Comm comm)
{
  // Get default linear algebra factory
  DefaultFactory factory;

  // Get list of available methods and preconditioners
  const std::map<std::string, std::string>
    methods = factory.krylov_solver_methods();
  const std::map<std::string, std::string>
    preconditioners = factory.krylov_solver_preconditioners();

  // Check that method is available
  if (!LinearSolver::in_list(method, methods))
  {
    dolfin_error("KrylovSolver.cpp",
                 "solve linear system using Krylov iteration",
                 "Unknown Krylov method \"%s\". "
                 "Use list_krylov_solver_methods() to list available Krylov methods",
                 method.c_str());
  }

  // Check that preconditioner is available
  if (!LinearSolver::in_list(preconditioner, preconditioners))
  {
    dolfin_error("KrylovSolver.cpp",
                 "solve linear system using Krylov iteration",
                 "Unknown preconditioner \"%s\". "
                 "Use list_krylov_solver_preconditioners() to list available preconditioners()",
                 preconditioner.c_str());
  }

  // Set default parameters
  parameters = dolfin::parameters("krylov_solver");

  // Initialize solver
  solver = factory.create_krylov_solver(comm, method, preconditioner);
  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
