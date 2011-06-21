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
#include "Form.h"
#include "VariationalProblem.h"
#include "LinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearVariationalSolver::
LinearVariationalSolver(const Form& a,
                        const Form& L,
                        Function& u,
                        std::vector<const BoundaryCondition*> bcs)
  : a(reference_to_no_delete_pointer(a)),
    L(reference_to_no_delete_pointer(L)),
    u(reference_to_no_delete_pointer(u))
{
  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); ++i)
    this->bcs.push_back(reference_to_no_delete_pointer(*bcs[i]));

  // Set parameters
  parameters = default_parameters();

  // Check forms
  check_forms();
}
//-----------------------------------------------------------------------------
LinearVariationalSolver::
LinearVariationalSolver(boost::shared_ptr<const Form> a,
                        boost::shared_ptr<const Form> L,
                        boost::shared_ptr<Function> u,
                        std::vector<boost::shared_ptr<const BoundaryCondition> > bcs)
  : a(a), L(L), u(u)
{
  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); ++i)
    this->bcs.push_back(bcs[i]);

  // Set parameters
  parameters = default_parameters();

  // Check forms
  check_forms();
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

  // Create matrix and vector
  boost::scoped_ptr<GenericMatrix> A(u->vector().factory().create_matrix());
  boost::scoped_ptr<GenericVector> b(u->vector().factory().create_vector());

  // Different assembly depending on whether or not the system is symmetric
  if (symmetric)
  {
    // Need to cast to DirichletBC to use assemble_system
    std::vector<const DirichletBC*> _bcs;
    for (uint i = 0; i < bcs.size(); i++)
    {
      const DirichletBC* _bc = dynamic_cast<const DirichletBC*>(bcs[i].get());
      if (!_bc)
        dolfin_error("LinearVariationalSolver.cpp",
                     "apply boundary condition in linear variational solver",
                     "only Dirichlet boundary conditions may be used for symmetric systems");
      _bcs.push_back(_bc);
    }

    // Assemble linear system and apply boundary conditions
    assemble_system(*A, *b,
                    *a, *L,
                    _bcs,
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
    for (uint i = 0; i < bcs.size(); i++)
      bcs[i]->apply(*A, *b);
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
    solver.solve(*A, u->vector(), *b);
  }
  else
  {
    KrylovSolver solver(solver_type, pc_type);
    solver.parameters.update(parameters("krylov_solver"));
    solver.solve(*A, u->vector(), *b);
  }

  end();
}
//-----------------------------------------------------------------------------
void LinearVariationalSolver::check_forms() const
{
  // Check rank of bilinear form a
  if (a->rank() != 2)
    dolfin_error("LinearVariationalSolver.cpp",
                 "create linear variational solver for a(u, v) == L(v) for all v",
                 "expecting the left-hand side to be a bilinear form (not rank %d).",
                 a->rank());

  // Check rank of linear form L
  if (L->rank() != 1)
    dolfin_error("LinearVariationalSolver.cpp",
                 "create linear variational solver for a(u, v) = L(v) for all v",
                 "expecting the right-hand side to be a linear form (not rank %d).",
                 a->rank());
}
//-----------------------------------------------------------------------------
