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
// Last changed: 2011-03-11

#include <dolfin/function/Function.h>
#include <dolfin/la/LinearAlgebraFactory.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include "assemble.h"
#include "VariationalProblem.h"
#include "LinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LinearVariationalSolver::solve(Function& u,
                                    const VariationalProblem& problem,
                                    const Parameters& parameters)

{
  begin("Solving linear variational problem.");

  // Get parameters
  std::string solver_type   = parameters["linear_solver"];
  const std::string pc_type = parameters["preconditioner"];
  const bool print_rhs      = parameters["print_rhs"];
  const bool symmetric      = problem.parameters["symmetric"];
  const bool print_matrix   = parameters["print_matrix"];

  // Create matrix and vector
  boost::scoped_ptr<GenericMatrix> A(u.vector().factory().create_matrix());
  boost::scoped_ptr<GenericVector> b(u.vector().factory().create_vector());

  // Get bilinear and linear forms
  boost::shared_ptr<const Form> a = problem.bilinear_form();
  boost::shared_ptr<const Form> L = problem.linear_form();
  assert(a);
  assert(L);


  // Different assembly depending on whether or not the system is symmetric
  if (symmetric)
  {
    // Need to cast to DirichletBC to use assemble_system
    std::vector<const DirichletBC*> bcs;
    const std::vector<boost::shared_ptr<const BoundaryCondition> > _bcs = problem.bcs();
    for (uint i = 0; i < _bcs.size(); i++)
    {
      const DirichletBC* bc = dynamic_cast<const DirichletBC*>(_bcs[i].get());
      if (!bc)
        error("Only Dirichlet boundary conditions may be used for assembly of symmetric system.");
      bcs.push_back(bc);
    }

    // Assemble linear system and apply boundary conditions
    assemble_system(*A, *b,
                    *a,
                    *L,
                    bcs,
                    0, 0, 0,
                    0,
                    true, false);
  }
  else
  {
    // Assemble linear system
    assemble(*A, *a);
    assemble(*b, *L);

    // Apply boundary conditions
    for (uint i = 0; i < problem.bcs().size(); i++)
      problem.bcs()[i]->apply(*A, *b);
  }

  // Print vector/matrix
  if (print_rhs)
    info(*b, true);
  if (print_matrix)
    info(*A, true);

  // Adjust solver type if necessary
  if (solver_type == "iterative")
  {
    if (symmetric)
      solver_type = "cg";
    else
      solver_type = "gmres";
  }

  // Solve linear system
  if (solver_type == "lu" || solver_type == "direct")
  {
    LUSolver solver;
    solver.parameters.update(parameters("lu_solver"));
    solver.solve(*A, u.vector(), *b);
  }
  else
  {
    KrylovSolver solver(solver_type, pc_type);
    solver.parameters.update(parameters("krylov_solver"));
    solver.solve(*A, u.vector(), *b);
  }

  end();
}
//-----------------------------------------------------------------------------
