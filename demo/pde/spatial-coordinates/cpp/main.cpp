// Copyright (C) 2010 Kristian B. Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-03-05
// Last changed: 2010-03-05
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
  bool inside(const double* x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  // Create mesh and function space
  UnitSquare mesh(32, 32);
  SpatialCoordinates::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational problem
  SpatialCoordinates::BilinearForm a(V, V);
  SpatialCoordinates::LinearForm L(V);

  // Compute solution
  VariationalProblem problem(a, L, bc);
  problem.parameters["linear_solver"] = "iterative";
  Function u(V);
  problem.solve(u);

  // Save solution in VTK format
  File file("spatial-coordinates.pvd");
  file << u;

  // Plot solution
  plot(u);

  return 0;
}
