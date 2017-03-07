// Copyright (C) 2010 Garth N. Wells
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

#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include "DefaultFactory.h"
#include "LinearSolver.h"
#include "LUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LUSolver::LUSolver(MPI_Comm comm, std::string method)
{
  init(comm, method);
}
//-----------------------------------------------------------------------------
LUSolver::LUSolver(std::string method) : LUSolver(MPI_COMM_WORLD, method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LUSolver::LUSolver(MPI_Comm comm,
                   std::shared_ptr<const GenericLinearOperator> A,
                   std::string method)
{
  // Initialize solver
  init(comm, method);

  // Set operator
  set_operator(A);
}
//-----------------------------------------------------------------------------
LUSolver::LUSolver(std::shared_ptr<const GenericLinearOperator> A,
                   std::string method) : LUSolver(MPI_COMM_WORLD, A, method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
LUSolver::~LUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LUSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  dolfin_assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
std::size_t LUSolver::solve(GenericVector& x, const GenericVector& b)
{
  dolfin_assert(solver);

  Timer timer("LU solver");
  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
std::size_t LUSolver::solve(const GenericLinearOperator& A, GenericVector& x,
                            const GenericVector& b)
{
  dolfin_assert(solver);

  Timer timer("LU solver");
  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
void LUSolver::init(MPI_Comm comm, std::string method)
{
  // Get default linear algebra factory
  DefaultFactory factory;

  // Get list of available methods
  std::map<std::string, std::string> methods = factory.lu_solver_methods();

  // Check that method is available
  if (!LinearSolver::in_list(method, methods))
  {
    dolfin_error("LUSolver.cpp",
                 "solve linear system using LU factorization",
                 "Unknown LU method \"%s\". "
                 "Use list_lu_solver_methods() to list available LU methods",
                 method.c_str());
  }

  // Set default parameters
  parameters = dolfin::parameters("lu_solver");

  // Initialize solver
  solver = factory.create_lu_solver(comm, method);
  solver->parameters.update(parameters);
}
//-----------------------------------------------------------------------------
