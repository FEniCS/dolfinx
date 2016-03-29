// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells
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
// Modified by Marie E. Rognes, 2011.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem.cpp)
// Last changed: 2012-07-30

#include <dolfin/common/NoDeleter.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/LinearSolver.h>
#include "Assembler.h"
#include "SystemAssembler.h"
#include "assemble.h"
#include "DirichletBC.h"
#include "Form.h"
#include "LinearVariationalProblem.h"
#include "LinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearVariationalSolver::
LinearVariationalSolver(std::shared_ptr<LinearVariationalProblem> problem)
  : _problem(problem)
{
  // Set parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
void LinearVariationalSolver::solve()
{
  begin("Solving linear variational problem.");

  // Get parameters
  std::string solver_type   = parameters["linear_solver"];
  const std::string pc_type = parameters["preconditioner"];
  const bool print_rhs      = parameters["print_rhs"];
  const bool symmetric      = parameters["symmetric"];
  const bool print_matrix   = parameters["print_matrix"];

  // Get problem data
  dolfin_assert(_problem);
  const auto a = _problem->bilinear_form();
  const auto L = _problem->linear_form();
  auto u = _problem->solution();
  auto bcs = _problem->bcs();

  dolfin_assert(a);
  dolfin_assert(L);
  dolfin_assert(u);

  // Create matrix and vector
  dolfin_assert(u->vector());
  MPI_Comm comm = u->vector()->mpi_comm();
  std::shared_ptr<GenericMatrix> A = u->vector()->factory().create_matrix();
  std::shared_ptr<GenericVector> b = u->vector()->factory().create_vector(comm);

  // Different assembly depending on whether or not the system is symmetric
  if (symmetric)
  {
    // Check that rhs (L) is not empty
    if (!L->ufc_form())
    {
      dolfin_error("LinearVariationalSolver.cpp",
                   "symmetric assembly in linear variational solver",
                   "Empty linear forms cannot be used with symmetric assembly");
    }

    // Assemble linear system and apply boundary conditions
    SystemAssembler assembler(a, L, bcs);
    assembler.assemble(*A, *b);
  }
  else
  {
    // Assemble linear system
    assemble(*A, *a);
    if (L->ufc_form())
      assemble(*b, *L);
    else
    {
      if (L->num_coefficients() != 0)
      {
        dolfin_error("LinearVariationalSolver.cpp",
                     "assemble linear form in linear variational solver",
                     "Empty linear forms cannot have coefficient");
      }
      A->init_vector(*b, 0);
    }

    // Apply boundary conditions
    for (std::size_t i = 0; i < bcs.size(); i++)
    {
      dolfin_assert(bcs[i]);
      bcs[i]->apply(*A, *b);
    }
  }

  // Print vector/matrix
  if (print_rhs)
    info(*b, true);
  if (print_matrix)
    info(*A, true);

  // Get list of available methods
  std::map<std::string, std::string>
    lu_methods = u->vector()->factory().lu_solver_methods();
  std::map<std::string, std::string>
    krylov_methods = u->vector()->factory().krylov_solver_methods();
  std::map<std::string, std::string>
    preconditioners = u->vector()->factory().krylov_solver_preconditioners();

  // Choose linear solver
  if (solver_type == "direct" || solver_type == "lu"
      || LinearSolver::in_list(solver_type, lu_methods))
  {
    std::string lu_method;
    if (solver_type == "direct" || solver_type == "lu")
      lu_method = "default";
    else
      lu_method = solver_type;

    // Solve linear system
    LUSolver solver(lu_method);
    solver.parameters.update(parameters("lu_solver"));
    solver.parameters["symmetric"] = (bool) parameters["symmetric"];
    solver.solve(*A, *u->vector(), *b);
  }
  else
  {
    if (solver_type == "iterative" || solver_type == "krylov")
    {
      // Adjust iterative solver type
      if (symmetric)
        solver_type = "cg";
      else
        solver_type = "gmres";
    }

    if (!LinearSolver::in_list(solver_type, krylov_methods))
    {
      dolfin_error("LinearVariationalSolver.cpp",
                   "solve linear system",
                   "Unknown solver method \"%s\". "
                   "Use list_linear_solver_methods() to list available methods",
                   solver_type.c_str());
    }

    if (pc_type != "default"
        && !LinearSolver::in_list(pc_type, preconditioners))
    {
      dolfin_error("LinearVariationalSolver.cpp",
                   "solve linear system",
                   "Unknown preconditioner method \"%s\". "
                   "Use list_krylov_solver_preconditioners() to list available methods",
                   pc_type.c_str());
    }

    // Solve linear system
    KrylovSolver solver(solver_type, pc_type);
    solver.parameters.update(parameters("krylov_solver"));
    solver.solve(*A, *u->vector(), *b);
  }

  end();
}
//-----------------------------------------------------------------------------
