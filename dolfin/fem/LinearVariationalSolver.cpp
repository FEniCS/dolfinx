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
#include "AssemblerTools.h"
#include "assemble.h"
#include "DirichletBC.h"
#include "PeriodicBC.h"
#include "Form.h"
#include "LinearVariationalProblem.h"
#include "LinearVariationalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearVariationalSolver::
LinearVariationalSolver(LinearVariationalProblem& problem)
  : problem(reference_to_no_delete_pointer(problem))
{
  // Set parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
LinearVariationalSolver::
LinearVariationalSolver(boost::shared_ptr<LinearVariationalProblem> problem)
  : problem(problem)
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
  dolfin_assert(problem);
  boost::shared_ptr<const Form> a(problem->bilinear_form());
  boost::shared_ptr<const Form> L(problem->linear_form());
  boost::shared_ptr<Function> u(problem->solution());
  std::vector<boost::shared_ptr<const BoundaryCondition> > bcs(problem->bcs());

  dolfin_assert(a);
  dolfin_assert(L);
  dolfin_assert(u);

  // Create matrix and vector
  dolfin_assert(u->vector());
  boost::shared_ptr<GenericMatrix> A = u->vector()->factory().create_matrix();
  boost::shared_ptr<GenericVector> b = u->vector()->factory().create_vector();

  // Different assembly depending on whether or not the system is symmetric
  if (symmetric)
  {
    // Check that rhs (L) is not empty
    if (!L->ufc_form())
    {
      dolfin_error("LinearVariationalSolver.cpp",
                   "symmetric assembly in linear variational solver",
                   "Empty linear forms cannot be used with symmetric assmebly");
    }

    // Need to cast to DirichletBC to use assemble_system
    std::vector<const DirichletBC*> _bcs;
    for (uint i = 0; i < bcs.size(); i++)
    {
      dolfin_assert(bcs[i]);
      const DirichletBC* _bc = dynamic_cast<const DirichletBC*>(bcs[i].get());
      if (!_bc)
      {
        dolfin_error("LinearVariationalSolver.cpp",
                     "apply boundary condition in linear variational solver",
                     "Only Dirichlet boundary conditions may be used for symmetric systems");
      }
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
    // Check for any periodic bcs
    typedef std::pair<dolfin::uint, dolfin::uint> DofOwnerPair;
    typedef std::pair<DofOwnerPair, DofOwnerPair> MasterSlavePair;
    std::vector<MasterSlavePair> dof_pairs;
    for (uint i = 0; i < bcs.size(); i++)
    {
      dolfin_assert(bcs[i]);
      const PeriodicBC* _bc = dynamic_cast<const PeriodicBC*>(bcs[i].get());
      if (_bc)
      {
        if (!dof_pairs.empty())
        {
          dolfin_error("LinearVariationalSolver.cpp",
                       "extract periodic boundary conditions in linear variational solver",
                       "Cannot have more than one PeriodicBC object");
        }
        _bc->compute_dof_pairs(dof_pairs);
      }
    }

    // Intialise matrix, taking into account periodic dofs
    AssemblerTools::init_global_tensor(*A, *a, dof_pairs, true, false, false);

    // Assemble linear system
    assemble(*A, *a, false);
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
      A->resize(*b, 0);
    }

    // Apply boundary conditions
    for (uint i = 0; i < bcs.size(); i++)
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

  // Adjust solver type if necessary
  if (solver_type == "iterative")
  {
    if (symmetric)
      solver_type = "cg";
    else
      solver_type = "gmres";
  }

  // Solve linear system
  LinearSolver solver(solver_type, pc_type);
  dolfin_assert(u->vector());
  solver.solve(*A, *u->vector(), *b);

  end();
}
//-----------------------------------------------------------------------------
