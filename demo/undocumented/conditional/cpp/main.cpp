// Copyright (C) 2010 Kristian B. Oelgaard
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
// First added:  2010-07-23
// Last changed: 2010-07-22
//
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//    f(x, y) =    -1.0 if (x - 0.33)^2 + (y - 0.67)^2 < 0.015
//                  5.0 if 0.015 < (x - 0.33)^2 + (y - 0.67)^2 < 0.025
//                 -1.0 if (x,y) in triangle( (0.55, 0.05), (0.95, 0.45), (0.55, 0.45) )
//                  0.0 otherwise
//
// and homogeneous Dirichlet boundary conditions.

#include <dolfin.h>
#include "Conditional.h"

using namespace dolfin;

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

int main()
{
  // Create mesh and function space
  UnitSquare mesh(64, 64);
  Conditional::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational problem
  Conditional::BilinearForm a(V, V);
  Conditional::LinearForm L(V);

  // Compute solution
  VariationalProblem problem(a, L, bc);

  Function u(V);
  problem.solve(u);

  // Save solution in VTK format
  File file("conditional.pvd");
  file << u;

  // Plot solution
  plot(u);

  return 0;
}
