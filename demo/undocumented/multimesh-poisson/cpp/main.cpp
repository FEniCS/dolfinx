// Copyright (C) 2013 Anders Logg
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
// Last changed: 2014-05-23
//
// This demo program solves MultiMeshPoisson's equation using a Cut and
// Composite Finite Element Method (MultiMesh) on a domain defined by
// three overlapping and non-matching meshes.

#include <dolfin.h>
#include "MultiMeshPoisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 2*x[0]*(1 - x[0]) + 2*x[1]*(1 - x[1]);
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

int main(int argc, char* argv[])
{
  // Increase log level
  set_log_level(DBG);

  // Don't reorder dofs (simplifies debugging)
  parameters["reorder_dofs_serial"] = false;

  // Create meshes
  int N = 32;
  UnitSquareMesh square(N, N);
  RectangleMesh rectangle_1(0.25, 0.25, 0.75, 0.75, N, N);

  // Rotate overlapping mesh
  rectangle_1.rotate(45);

  // Create function spaces
  MultiMeshPoisson::FunctionSpace V0(square);
  MultiMeshPoisson::FunctionSpace V1(rectangle_1);
  //MultiMeshPoisson::FunctionSpace V2(rectangle_2);

  // Some of this stuff may be wrapped or automated later to avoid
  // needing to explicitly call add() and build()

  // Create forms
  MultiMeshPoisson::BilinearForm a0(V0, V0);
  MultiMeshPoisson::BilinearForm a1(V1, V1);
  //MultiMeshPoisson::BilinearForm a2(V2, V2);
  MultiMeshPoisson::LinearForm L0(V0);
  MultiMeshPoisson::LinearForm L1(V1);
  //MultiMeshPoisson::LinearForm L2(V2);

  // Build MultiMesh function space
  MultiMeshFunctionSpace V;
  V.parameters("multimesh")["quadrature_order"] = 2;
  V.add(V0);
  V.add(V1);
  //V.add(V2);
  V.build();

  // Set coefficients
  Source f;
  L0.f = f;
  L1.f = f;
  //L2.f = f;

  // FIXME: Testing
  Parameters p;
  p.add("c0", 1.0);
  p.add("c1", 0.0); // For some reason, including this term gives a kink in the solution
  p.add("c2", 1.0);
  p.add("c3", 1.0);
  p.parse(argc, argv);
  a0.c0 = static_cast<double>(p["c0"]);
  a0.c1 = static_cast<double>(p["c1"]);
  a0.c2 = static_cast<double>(p["c2"]);
  a0.c3 = static_cast<double>(p["c3"]);
  a1.c0 = static_cast<double>(p["c0"]);
  a1.c1 = static_cast<double>(p["c1"]);
  a1.c2 = static_cast<double>(p["c2"]);
  a1.c3 = static_cast<double>(p["c3"]);

  // Build MultiMesh forms
  MultiMeshForm a(V, V);
  a.add(a0);
  a.add(a1);
  //a.add(a2);
  a.build();
  MultiMeshForm L(V);
  L.add(L0);
  L.add(L1);
  //L.add(L2);
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
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  u0_file << *u.part(0);
  u1_file << *u.part(1);

  // Plot solution
  plot(V.multimesh());
  plot(u.part(0), "u_0");
  plot(u.part(1), "u_1");
  //plot(u.part(2), "u_2");
  interactive();

  return 0;
}
