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
// Last changed: 2013-09-12
//
// This demo program solves Poisson's equation using a Cut and
// Composite Finite Element Method (CCFEM) on a domain defined by
// three overlapping and non-matching meshes: a mesh of two displaced
// unit circles which partly overlap a mesh of a unit square and each
// other.

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

int main()
{
  info("THIS DEMO IS WORK IN PROGRESS!");

  // Increase log level
  set_log_level(PROGRESS);

  // Create meshes
  UnitSquareMesh square(8, 8);
  UnitCircleMesh circle_1(8);
  UnitCircleMesh circle_2(8);

  // Displace circle meshes
  Point dx(0.5, 0.5);
  circle_1.translate(dx);
  circle_2.translate(-dx);

  // Create function spaces
  Poisson::FunctionSpace V0(square);
  Poisson::FunctionSpace V1(circle_1);
  Poisson::FunctionSpace V2(circle_2);

  // Create forms
  Poisson::BilinearForm a0(V0, V0);
  Poisson::BilinearForm a1(V1, V1);
  Poisson::BilinearForm a2(V2, V2);

  // Build CCFEM function space
  CCFEMFunctionSpace V;
  V.add(V0);
  V.add(V1);
  V.add(V2);
  V.build();

  // Build CCFEM form
  CCFEMForm a(V);
  a.add(a0);
  a.add(a1);
  a.add(a2);
  a.build();

  // Assemble linear system
  Matrix A;
  CCFEMAssembler assembler;
  assembler.assemble(A, a);

  return 0;
}
