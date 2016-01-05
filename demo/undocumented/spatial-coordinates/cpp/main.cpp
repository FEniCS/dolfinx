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
// First added:  2010-03-05
// Last changed: 2012-11-12
//
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// and boundary conditions given by
//
//     u(x, y) = 0        for x = 0 or x = 1
// du/dn(x, y) = -sin(5*x) for y = 0 or y = 1
//
// This demo is identical to the Poisson demo with the only difference that
// the source and flux term is expressed using SpatialCoordinates in the
// variational formulation.

#include <dolfin.h>
#include "SpatialCoordinates.h"

using namespace dolfin;

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  // Create mesh and function space
  UnitSquareMesh mesh(32, 32);
  SpatialCoordinates::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  auto bc = std::make_shared<DirichletBC>(V, u0, boundary);

  // Define variational problem
  auto a = std::make_shared<SpatialCoordinates::BilinearForm>(V, V);
  auto L = std::make_shared<SpatialCoordinates::LinearForm>(V);
  auto u = std::make_shared<Function>(V);
  LinearVariationalProblem problem(a, L, u, {bc});

  // Compute solution
  LinearVariationalSolver solver(problem);
  solver.parameters["linear_solver"] = "iterative";
  solver.solve();

  // Save solution in VTK format
  File file("spatial-coordinates.pvd");
  file << *u;

  // Plot solution
  plot(*u);
  interactive();

  return 0;
}
