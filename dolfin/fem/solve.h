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
// Last changed: 2011-06-22

#ifndef __SOLVE_FEM_H
#define __SOLVE_FEM_H

#include <vector>

namespace dolfin
{

  // Forward declarations
  class Equation;
  class Function;
  class BoundaryCondition;

  /// Solve variational problem a == L or F == 0
  void solve(const Equation& equation,
             Function& u,
             std::vector<const BoundaryCondition*> bcs);

}

#endif
