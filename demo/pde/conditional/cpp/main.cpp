// Copyright (C) 2010 Kristian B. Ølgaard (k.b.oelgaard@gmail.com).
// Licensed under the GNU GPL version 3 or any later version
//
// First added:  2010-07-23
// Last changed: 2010-07-23
//
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 1.0 if (x - 0.33)^2 + (y - 0.67)^2 < 0.025, 0.0 otherwise
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
  UnitSquare mesh(32, 32);
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
