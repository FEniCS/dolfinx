// Copyright (C) 2011 Marie E. Rognes
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
// First added:  2011-06-30
// Last changed: 2011-06-30

#include <dolfin/adaptivity/AdaptiveLinearVariationalSolver.h>
#include <dolfin/adaptivity/AdaptiveNonlinearVariationalSolver.h>

#include "LinearVariationalProblem.h"
#include "NonlinearVariationalProblem.h"
#include "Equation.h"
#include "adaptivesolve.h"

//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u,
                   const double tol,
                   GoalFunctional& M)
{
  // Create empty list of boundary conditions
  std::vector<const BoundaryCondition*> bcs;

  // Call common adaptive solve function
  solve(equation, u, bcs, tol, M);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u,
                   const BoundaryCondition& bc,
                   const double tol,
                   GoalFunctional& M)
{
  // Create list containing single boundary condition
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bc);

  // Call common adaptive solve function
  solve(equation, u, bcs, tol, M);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation,
                   Function& u,
                   std::vector<const BoundaryCondition*> bcs,
                   const double tol,
                   GoalFunctional& M)

{
  // Solve linear problem
  if (equation.is_linear())
  {
    LinearVariationalProblem problem(*equation.lhs(), *equation.rhs(), u, bcs);
    AdaptiveLinearVariationalSolver solver(problem);
    solver.solve(tol, M);
  } else
  {
    dolfin_error("solve.cpp",
                 "solve nonlinear variational problem adaptively",
                 "solve not implemented without Jacobian");
  }
}
//-----------------------------------------------------------------------------
