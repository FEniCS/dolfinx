// Copyright (C) 2011 Anders Logg
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
// First added:  2011-06-22
// Last changed: 2011-09-22

#include "LinearVariationalProblem.h"
#include "LinearVariationalSolver.h"
#include "NonlinearVariationalProblem.h"
#include "NonlinearVariationalSolver.h"
#include "Equation.h"
#include "solve.h"

//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u, Parameters params)
{
  // Create empty list of boundary conditions
  std::vector<const BoundaryCondition*> bcs;

  // Call common solve function
  solve(equation, u, bcs, params);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u,
                   const BoundaryCondition& bc, 
		   Parameters params)
{
  // Create list containing single boundary condition
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bc);

  // Call common solve function
  solve(equation, u, bcs, params);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u,
                   std::vector<const BoundaryCondition*> bcs, 
		   Parameters params)
{
  // Solve linear problem
  if (equation.is_linear())
  {
    LinearVariationalProblem problem(*equation.lhs(), *equation.rhs(), u, bcs);
    LinearVariationalSolver solver(problem);
    solver.parameters.update(params);
    solver.solve();
  }

  // Solve nonlinear problem
  else
  {
    NonlinearVariationalProblem problem(*equation.lhs(), u, bcs);
    NonlinearVariationalSolver solver(problem);
    solver.parameters.update(params);
    solver.solve();
  }
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u,
                   const Form& J, 
		   Parameters params)
{
  // Create empty list of boundary conditions
  std::vector<const BoundaryCondition*> bcs;

  // Call common solve function
  solve(equation, u, bcs, J, params);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u,
                   const BoundaryCondition& bc,
                   const Form& J, 
		   Parameters params)
{
  // Create list containing single boundary condition
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bc);

  // Call common solve function
  solve(equation, u, bcs, J, params);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u,
                   std::vector<const BoundaryCondition*> bcs,
                   const Form& J, 
		   Parameters params)
{
  // Check that the problem is linear
  if (equation.is_linear())
    dolfin_error("solve.cpp",
                 "solve nonlinear variational problem",
                 "Variational problem is linear");

  // Solve nonlinear problem
  NonlinearVariationalProblem problem(*equation.lhs(), u, bcs, J);
  NonlinearVariationalSolver solver(problem);
  solver.parameters.update(params);
  solver.solve();
}
//-----------------------------------------------------------------------------
