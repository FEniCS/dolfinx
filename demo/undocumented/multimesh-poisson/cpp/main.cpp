// Copyright (C) 2013-2014 Anders Logg
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
// Last changed: 2014-06-15
//
// This demo program solves Poisson's equation on a domain defined by
// three overlapping and non-matching meshes.

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

// Files for storing solution
File u0_file("u0.pvd");
File u1_file("u1.pvd");
File u2_file("u2.pvd");

// Compute solution for given mesh configuration
void solve(double x1, double y1, double x2, double y2, bool plot_solution)
{
  // Create meshes
  int N = 16;
  Point p1(x1+10, y1+10);
  double r = 0.5;
  double h = 0.1;
  RectangleMesh mesh_0(-r, -r, r, r, N, N);
  RectangleMesh mesh_1(x2 - r, y2 - r, x2 + r, y2 + r, N, N);
  CircleMesh mesh_2(p1, r, h);

  // Rotate overlapping mesh
  //rectangle_1.rotate(45);

  // Create function spaces
  MultiMeshPoisson::FunctionSpace V0(mesh_0);
  MultiMeshPoisson::FunctionSpace V1(mesh_1);
  MultiMeshPoisson::FunctionSpace V2(mesh_2);

  // FIXME: Some of this stuff may be wrapped or automated later to
  // avoid needing to explicitly call add() and build()

  // Create forms
  MultiMeshPoisson::BilinearForm a0(V0, V0);
  MultiMeshPoisson::BilinearForm a1(V1, V1);
  MultiMeshPoisson::BilinearForm a2(V2, V2);
  MultiMeshPoisson::LinearForm L0(V0);
  MultiMeshPoisson::LinearForm L1(V1);
  MultiMeshPoisson::LinearForm L2(V2);

  // Build multimesh function space
  MultiMeshFunctionSpace V;
  V.parameters("multimesh")["quadrature_order"] = 2;
  V.add(V0);
  V.add(V1);
  V.add(V2);
  V.build();

  // Set coefficients
  Source f;
  L0.f = f;
  L1.f = f;
  L2.f = f;

  // Build multimesh forms
  MultiMeshForm a(V, V);
  MultiMeshForm L(V);
  a.add(a0);
  a.add(a1);
  a.add(a2);
  L.add(L0);
  L.add(L1);
  L.add(L2);
  a.build();
  L.build();

  // Create boundary condition
  Constant zero(0);
  DirichletBoundary boundary;
  MultiMeshDirichletBC bc(V, zero, boundary);

  // Assemble linear system
  Matrix A;
  Vector b;
  MultiMeshAssembler assembler;
  assembler.assemble(A, a);
  assembler.assemble(b, L);

  // Apply boundary condition
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

  // FIXME: Testing
  //  set_log_level(DBG);
  parameters["reorder_dofs_serial"] = false;

  // Parameters
  const double T = 40.0;
  const int N = 400;
  const double dt = T / N;

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
    solve(x1, y1, x2, y2, n == N - 1);
  }

  return 0;
}
