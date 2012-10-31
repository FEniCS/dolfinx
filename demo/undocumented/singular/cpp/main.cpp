// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-10-31
// Last changed:
//
// This demo program illustrates how to solve Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with pure Neumann boundary conditions:
//
//     du/dn(x, y) = -sin(5*x)
//
// and source f given by
//
//     f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// Since only Neumann conditions are applied, u is only determined up to
// a constant c by the above equations. An addition constraint is thus
// required, for instance
//
//   \int u = 0
//
// This is accomplished in this demo by using a Krylov iterative solver
// that removes the component in the null space from the solution vector.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Boundary flux (Neumann boundary condition)
class Flux : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -sin(5*x[0]);
  }
};

int main()
{
  // Create mesh and function space
  UnitSquare mesh(64, 64);
  Poisson::FunctionSpace V(mesh);

  // Define variational problem
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  Source f;
  Flux g;
  L.f = f;
  L.g = g;

  // Assemble system
  boost::shared_ptr<GenericMatrix> A(new Matrix);
  Vector b;
  assemble(*A, a);
  assemble(b, L);

  // Solution Function
  Function u(V);

  // Create Krylov solver
  KrylovSolver solver(A, "gmres");

  // Create null space basis and attach to Krylov solver
  Vector null_space0(*u.vector());
  V.dofmap()->set(null_space0, 1.0);
  std::vector<const GenericVector*> null_space;
  null_space.push_back(&null_space0);
  solver.set_nullspace(null_space);

  // Solve
  solver.solve(*u.vector(), b);

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
