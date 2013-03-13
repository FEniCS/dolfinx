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

#ifndef __SOLVE_FEM_H
#define __SOLVE_FEM_H

#include <vector>
#include <dolfin/parameter/Parameters.h>

namespace dolfin
{

  // Forward declarations
  class Equation;
  class Function;
  class DirichletBC;
  class Form;

  //--- Linear / nonlinear problems (no Jacobian specified) ---

  /// Solve linear variational problem a(u, v) == L(v) or nonlinear
  /// variational problem F(u; v) = 0 without boundary conditions.
  ///
  /// Optional parameters can be passed to the LinearVariationalSolver
  /// or NonlinearVariationalSolver classes.
  void solve(const Equation& equation,
             Function& u,
             Parameters parameters=empty_parameters);

  /// Solve linear variational problem a(u, v) == L(v) or nonlinear
  /// variational problem F(u; v) = 0 with a single boundary condition.
  ///
  /// Optional parameters can be passed to the LinearVariationalSolver
  /// or NonlinearVariationalSolver classes.
  void solve(const Equation& equation,
             Function& u,
             const DirichletBC& bc,
             Parameters parameters=empty_parameters);

  /// Solve linear variational problem a(u, v) == L(v) or nonlinear
  /// variational problem F(u; v) = 0 with a list of boundary conditions.
  ///
  /// Optional parameters can be passed to the LinearVariationalSolver
  /// or NonlinearVariationalSolver classes.
  void solve(const Equation& equation,
             Function& u,
             std::vector<const DirichletBC*> bcs,
             Parameters parameters=empty_parameters);

  //--- Nonlinear problems (Jacobian specified) ---

  /// Solve nonlinear variational problem F(u; v) == 0 without boundary
  /// conditions. The argument J should provide the Jacobian bilinear
  /// form J = dF/du.
  ///
  /// Optional parameters can be passed to the LinearVariationalSolver
  /// or NonlinearVariationalSolver classes.
  void solve(const Equation& equation,
             Function& u,
             const Form& J,
             Parameters parameters=empty_parameters);

  /// Solve nonlinear variational problem F(u; v) == 0 with a single
  /// boundary condition. The argument J should provide the Jacobian
  /// bilinear form J = dF/du.
  ///
  /// Optional parameters can be passed to the LinearVariationalSolver
  /// or NonlinearVariationalSolver classes.
  void solve(const Equation& equation,
             Function& u,
             const DirichletBC& bc,
             const Form& J,
             Parameters parameters=empty_parameters);

  /// Solve nonlinear variational problem F(u; v) == 0 with a list of
  /// boundary conditions. The argument J should provide the Jacobian
  /// bilinear form J = dF/du.
  ///
  /// Optional parameters can be passed to the LinearVariationalSolver
  /// or NonlinearVariationalSolver classes.
  void solve(const Equation& equation,
             Function& u,
             std::vector<const DirichletBC*> bcs,
             const Form& J,
             Parameters parameters=empty_parameters);

}

#endif
