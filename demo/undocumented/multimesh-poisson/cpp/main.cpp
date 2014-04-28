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
// Last changed: 2014-04-28
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
    const double x0 = 1.0;
    const double y0 = 1.0;
    double dx = x[0] - x0;
    double dy = x[1] - y0;
    values[0] = 100*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

int main()
{
  // Increase log level
  set_log_level(DBG);

  // Create meshes
  UnitSquareMesh square(32, 32);
  RectangleMesh rectangle_1(0.250, 0.250, 0.75, 0.75, 16, 16);
  //RectangleMesh rectangle_1(0.250, 0.250, 0.625, 0.625, 16, 16);
  //RectangleMesh rectangle_2(0.375, 0.375, 0.750, 0.750, 16, 16);

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

  // Set coefficients
  Source f;
  L0.f = f;
  L1.f = f;
  //L2.f = f;

  // Build MultiMesh function space
  MultiMeshFunctionSpace V;
  V.add(V0);
  V.add(V1);
  //V.add(V2);
  V.build();

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

  // Assemble linear system
  Matrix A;
  Vector b;
  MultiMeshAssembler assembler;
  assembler.assemble(A, a);
  assembler.assemble(b, L);

  // Lock inactive dofs
  //A.ident_zeros();

  // Compute solution
  MultiMeshFunction u(V);
  solve(A, *u.vector(), b);

  cout << endl;
  cout << "||x|| = " << u.vector()->norm("l2") << endl;

  plot(V.multimesh());
  plot(u.part(0), "u_0");
  plot(u.part(1), "u_1");
  //plot(u.part(2), "u_2");
  interactive();

  //cout << "A = " << endl; info(A, true);
  //cout << "b = " << endl; info(b, true);
  //cout << "x = " << endl; info(*u.vector(), true);

  return 0;
}
