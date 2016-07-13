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
// Last changed: 2016-03-02
//
// This demo program solves Poisson's equation on a domain defined by
// three overlapping and non-matching meshes. The solution is computed
// on a sequence of rotating meshes to test the multimesh
// functionality.

#include <cmath>
#include <dolfin.h>
#include "MultiMeshPoisson.h"

using namespace dolfin;
using std::make_shared;

void assemble_scalar(double x1, double y1,
                     double x2, double y2)
{
  // Create meshes
  double r = 0.5;
  auto mesh_0 = make_shared<RectangleMesh>(Point(-r, -r), Point(r, r), 16, 16);
  auto mesh_1 = make_shared<RectangleMesh>(Point(x1 - r, y1 - r), Point(x1 + r, y1 + r), 8, 8);
  auto mesh_2 = make_shared<RectangleMesh>(Point(x2 - r, y2 - r), Point(x2 + r, y2 + r), 8, 8);

  // Build multimesh
  auto multimesh = make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  multimesh->add(mesh_1);
  multimesh->add(mesh_2);
  multimesh->build();

  // The function v
  class MyFunction : public Expression
  {
  public:

    void eval(Array<double>& values, const Array<double>& x) const
    { values[0] = sin(x[0]) + cos(x[1]); }

  };


  // Create forms
  auto v = std::make_shared<MyFunction>();
  MultiMeshPoisson::MultiMeshFunctional M(multimesh);

  // Assemble functional
  double b = assemble_multimesh(M);
}

int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }

  // Compute solution
  assemble_scalar(0, 0, 1, 1);

  return 0;
}
