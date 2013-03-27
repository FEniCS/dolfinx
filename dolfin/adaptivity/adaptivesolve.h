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

#ifndef __ADAPTIVE_SOLVE_H
#define __ADAPTIVE_SOLVE_H

#include <vector>

namespace dolfin
{

  // Forward declarations
  class DirichletBC;
  class Equation;
  class Form;
  class Function;
  class GoalFunctional;

  //--- Adaptive solve of linear problems ---

  /// Solve linear variational problem a(u, v) == L(v) without
  /// essential boundary conditions
  void solve(const Equation& equation,
             Function& u,
             const double tol,
             GoalFunctional& M);

  /// Solve linear variational problem a(u, v) == L(v) with single
  /// boundary condition
  void solve(const Equation& equation,
             Function& u,
             const DirichletBC& bc,
             const double tol,
             GoalFunctional& M);

  /// Solve linear variational problem a(u, v) == L(v) with list of
  /// boundary conditions
  void solve(const Equation& equation,
             Function& u,
             std::vector<const DirichletBC*> bcs,
             const double tol,
             GoalFunctional& M);

  //--- Adaptive solve of nonlinear problems ---

  /// Solve nonlinear variational problem F(u; v) = 0 without
  /// essential boundary conditions
  void solve(const Equation& equation,
             Function& u,
             const Form& J,
             const double tol,
             GoalFunctional& M);

  /// Solve linear variational problem F(u; v) = 0 with single
  /// boundary condition
  void solve(const Equation& equation,
             Function& u,
             const DirichletBC& bc,
             const Form& J,
             const double tol,
             GoalFunctional& M);

  /// Solve linear variational problem F(u; v) = 0 with list of
  /// boundary conditions
  void solve(const Equation& equation,
             Function& u,
             std::vector<const DirichletBC*> bcs,
             const Form& J,
             const double tol,
             GoalFunctional& M);

}

#endif
