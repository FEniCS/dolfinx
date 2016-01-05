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
// Last changed: 2012-11-09

#include "LinearVariationalProblem.h"
#include "LinearVariationalSolver.h"
#include "NonlinearVariationalProblem.h"
#include "NonlinearVariationalSolver.h"
#include "Equation.h"
#include "solve.h"

//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation, Function& u, Parameters parameters)
{
  // Call common solve function
  solve(equation, u, std::vector<const DirichletBC*>(), parameters);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation, Function& u, const DirichletBC& bc,
		   Parameters parameters)
{
  // Call common solve function
  solve(equation, u, {&bc}, parameters);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation, Function& u,
                   std::vector<const DirichletBC*> bcs, Parameters parameters)
{
  // Pack bcs
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;
  for (auto bc : bcs)
    _bcs.push_back(reference_to_no_delete_pointer(*bc));

  // Solve linear problem
  if (equation.is_linear())
  {
    LinearVariationalProblem problem(equation.lhs(), equation.rhs(),
                                     reference_to_no_delete_pointer(u), _bcs);
    LinearVariationalSolver solver(problem);
    solver.parameters.update(parameters);
    solver.solve();
  }
  else
  {
    // Solve nonlinear problem
    NonlinearVariationalProblem problem(equation.lhs(),
                                        reference_to_no_delete_pointer(u),
                                        _bcs);
    NonlinearVariationalSolver solver(problem);
    solver.parameters.update(parameters);
    solver.solve();
  }
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation, Function& u, const Form& J,
		   Parameters parameters)
{
  // Call common solve function
  solve(equation, u, {}, J, parameters);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation, Function& u, const DirichletBC& bc,
                   const Form& J, Parameters parameters)
{
  // Call common solve function
  solve(equation, u, {&bc}, J, parameters);
}
//-----------------------------------------------------------------------------
void dolfin::solve(const Equation& equation, Function& u,
                   std::vector<const DirichletBC*> bcs, const Form& J,
		   Parameters parameters)
{
  // Check that the problem is linear
  if (equation.is_linear())
  {
    dolfin_error("solve.cpp",
                 "solve nonlinear variational problem",
                 "Variational problem is linear");
  }

  // Pack bcs
  std::vector<std::shared_ptr<const DirichletBC>> _bcs;
  for (auto bc : bcs)
    _bcs.push_back(reference_to_no_delete_pointer(*bc));

  // Solve nonlinear problem
  NonlinearVariationalProblem problem(equation.lhs(),
                                      reference_to_no_delete_pointer(u), _bcs,
                                      reference_to_no_delete_pointer(J));
  NonlinearVariationalSolver solver(problem);
  solver.parameters.update(parameters);
  solver.solve();
}
//-----------------------------------------------------------------------------
