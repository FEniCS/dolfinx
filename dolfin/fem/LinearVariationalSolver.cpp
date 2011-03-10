// Copyright (C) 2008-2011 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Marie E. Rognes, 2011.
//
// First added:  2011-01-14 (2008-12-26 as VariationalProblem.cpp)
// Last changed: 2011-01-17

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

  // Different assembly depending on whether or not the system is symmetric
  if (symmetric)
  {
    // Need to cast to DirichletBC to use assemble_system
    std::vector<const DirichletBC*> bcs;
    for (uint i = 0; i < problem.bcs().size(); i++)
    {
      const DirichletBC* bc = dynamic_cast<const DirichletBC*>(problem.bcs()[i]);
      if (!bc)
        error("Only Dirichlet boundary conditions may be used for assembly of symmetric system.");
      bcs.push_back(bc);
    }

    // Assemble linear system and apply boundary conditions
    assemble_system(*A, *b,
                    problem.bilinear_form(),
                    problem.linear_form(),
                    bcs,
                    problem.cell_domains(),
                    problem.exterior_facet_domains(),
                    problem.interior_facet_domains(),
                    0,
                    true, false);
  }
  else
  {
    // Assemble linear system
    std::cout << "Assemble LHS" << std::endl;
    assemble(*A,
             problem.bilinear_form(),
             problem.cell_domains(),
             problem.exterior_facet_domains(),
             problem.interior_facet_domains());
    std::cout << "Assemble RHS" << std::endl;
    assemble(*b,
             problem.linear_form(),
             problem.cell_domains(),
             problem.exterior_facet_domains(),
             problem.interior_facet_domains());

    std::cout << "Apply BCS" << std::endl;
    // Apply boundary conditions
    for (uint i = 0; i < problem.bcs().size(); i++)
      problem.bcs()[i]->apply(*A, *b);
    std::cout << "End BCS" << std::endl;
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
