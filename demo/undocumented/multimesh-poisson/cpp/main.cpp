// Copyright (C) 2013-2015 Anders Logg
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
// First added:  2013-06-26
// Last changed: 2015-11-16
//
// This demo program solves Poisson's equation on a domain defined by
// three overlapping and non-matching meshes. The solution is computed
// on a sequence of rotating meshes to test the multimesh
// functionality.

#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 1.0;
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

// Compute solution for given mesh configuration
void solve_poisson(double t,
                   double x1, double y1,
                   double x2, double y2,
                   bool plot_solution,
                   File& u0_file, File& u1_file, File& u2_file)
{
  // Create meshes
  double r = 0.5;
  RectangleMesh mesh_0(Point(-r, -r), Point(r, r), 16, 16);
  RectangleMesh mesh_1(Point(x1 - r, y1 - r), Point(x1 + r, y1 + r), 8, 8);
  RectangleMesh mesh_2(Point(x2 - r, y2 - r), Point(x2 + r, y2 + r), 8, 8);
  mesh_1.rotate(70*t);
  mesh_2.rotate(-70*t);

  // Build multimesh
  MultiMesh multimesh;
  multimesh.add(mesh_0);
  multimesh.add(mesh_1);
  multimesh.add(mesh_2);
  multimesh.build();

  // Create function space
  MultiMeshPoisson::MultiMeshFunctionSpace V(multimesh);

  // Create forms
  MultiMeshPoisson::MultiMeshBilinearForm a(V, V);
  MultiMeshPoisson::MultiMeshLinearForm L(V);

  // Attach coefficients
  Source f;
  L.f = f;

  // Assemble linear system
  Matrix A;
  Vector b;
  assemble_multimesh(A, a);
  assemble_multimesh(b, L);

  // Apply boundary condition
  Constant zero(0);
  DirichletBoundary boundary;
  MultiMeshDirichletBC bc(V, zero, boundary);
  bc.apply(A, b);

  // Compute solution
  MultiMeshFunction u(V);
  solve(A, *u.vector(), b);

  // Save to file
  u0_file << *u.part(0);
  u1_file << *u.part(1);
  u2_file << *u.part(2);

  // Plot solution (last time)
  if (plot_solution)
  {
    plot(V.multimesh());
    plot(u.part(0), "u_0");
    plot(u.part(1), "u_1");
    plot(u.part(2), "u_2");
    interactive();
  }
}

int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }

  // Parameters
  const double T = 40.0;
  const std::size_t N = 400;
  const double dt = T / N;

  // Files for storing solution
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");

  // Iterate over configurations
  for (std::size_t n = 0; n < N; n++)
  {
    info("Computing solution, step %d / %d.", n + 1, N);

    // Compute coordinates for meshes
    const double t = dt*n;
    const double x1 = sin(t)*cos(2*t);
    const double y1 = cos(t)*cos(2*t);
    const double x2 = cos(t)*cos(2*t);
    const double y2 = sin(t)*cos(2*t);

    // Compute solution
    solve_poisson(t, x1, y1, x2, y2, n == N - 1,
                  u0_file, u1_file, u2_file);
  }

  return 0;
}
