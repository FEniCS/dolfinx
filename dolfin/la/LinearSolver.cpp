// Copyright (C) 2008-2011 Anders Logg
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
// Modified by Johan Hake, 2010.
// Modified by Garth N. Wells, 2010.
//
// First added:  2008-05-10
// Last changed: 2011-10-19

#include "DefaultFactory.h"
#include "KrylovSolver.h"
#include "LUSolver.h"
#include "CholmodCholeskySolver.h"
#include "LinearSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearSolver::LinearSolver(std::string method,
                           std::string preconditioner)
{
  // Get default linear algebra factory
  DefaultFactory factory;

  // Get list of available methods
  std::vector<std::pair<std::string, std::string> >
    lu_methods = factory.lu_solver_methods();
  std::vector<std::pair<std::string, std::string> >
    krylov_methods = factory.krylov_solver_methods();

  // Handle some default and generic solver options
  if (method == "default")
    method = "lu";
  else if (method == "direct")
    method = "lu";
  else if (method == "iterative")
    method = "gmres";

  // Choose solver
  if (method == "lu" || in_list(method, lu_methods))
  {
    // Adjust preconditioner default --> none
    if (preconditioner == "default")
      preconditioner = "none";

    // Check that preconditioner has not been set
    if (preconditioner != "none")
    {
      dolfin_error("LinearSolver.cpp",
                   "solve linear system",
                   "Preconditioner may not be specified for LU solver");
    }

    // Use default LU method if "lu" is specified
    if (method == "lu")
      method = "default";

    // Initialize solver
    solver.reset(new LUSolver(method));
  }
  else if (method == "cholesky")
  {
    // Adjust preconditioner default --> none
    if (preconditioner == "default")
      preconditioner = "none";

    // Check that preconditioner has not been set
    if (preconditioner != "none")
    {
      dolfin_error("LinearSolver.cpp",
                   "solve linear system",
                   "Preconditioner may not be specified for Cholesky solver");
    }

    // Initialize solver
    solver.reset(new CholmodCholeskySolver());
  }
  else if (in_list(method, krylov_methods))
  {
    // Method and preconditioner will be checked by KrylovSolver

    // Initialize solver
    solver.reset(new KrylovSolver(method, preconditioner));
  }
  else
  {
    dolfin_error("LinearSolver.cpp",
                 "solve linear system",
                 "Unknown solver method \"%s\". "
                 "Use list_linear_solver_methods() to list available methods",
                 method.c_str());
  }

  // Get parameters
  parameters = solver->parameters;

  // Rename the parameters
  parameters.rename("linear_solver");
}
//-----------------------------------------------------------------------------
LinearSolver::~LinearSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LinearSolver::set_operator(const boost::shared_ptr<const GenericMatrix> A)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operator(A);
}
//-----------------------------------------------------------------------------
void LinearSolver::set_operators(const boost::shared_ptr<const GenericMatrix> A,
                                 const boost::shared_ptr<const GenericMatrix> P)
{
  assert(solver);
  solver->parameters.update(parameters);
  solver->set_operators(A, P);
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(const GenericMatrix& A, GenericVector& x,
                                 const GenericVector& b)
{
  assert(solver);
  //check_dimensions(A, x, b);

  solver->parameters.update(parameters);
  return solver->solve(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint LinearSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(solver);
  //check_dimensions(get_operator(), x, b);

  solver->parameters.update(parameters);
  return solver->solve(x, b);
}
//-----------------------------------------------------------------------------
bool
LinearSolver::in_list(const std::string& method,
                      const std::vector<std::pair<std::string, std::string> > methods)
{
  for (uint i = 0; i < methods.size(); i++)
    if (method == methods[i].first)
      return true;
  return false;
}
//-----------------------------------------------------------------------------
