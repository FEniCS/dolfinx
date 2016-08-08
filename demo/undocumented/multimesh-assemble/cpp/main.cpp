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
#include "MultiMeshAssemble.h"
#include "MeshAssemble.h"
#include "compute_volume.h"
using namespace dolfin;
using std::make_shared;

void assemble_scalar()
{
  // Create meshes
  auto mesh_0 = make_shared<RectangleMesh>(Point(0.,0.), Point(2., 2.), 16, 16);
  auto mesh_1 = make_shared<RectangleMesh>(Point(1., 1.), Point(3., 3.), 10, 10);
  auto mesh_2 = make_shared<RectangleMesh>(Point(-1., -1.),
					   Point(0.2, 0.2), 8, 8);

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
    //{ values[0] = sin(x[0]) + cos(x[1]); }
    { values[0] = 1; }

  };
  double p = 1;
  // Create function space
  auto V = make_shared<MultiMeshAssemble::MultiMeshFunctionSpace>(multimesh);

  // Create forms
  auto v = std::make_shared<MyFunction>();
  MultiMeshAssemble::MultiMeshFunctional M(multimesh);

  // Set MultiMeshFunctional coefficient
  M.q = v;

  // Compute the MultiMeshFunctional
  double b = assemble_multimesh(M);
  cout << "Area of the total mesh (Functional)" << endl;
  cout << b << endl;

  // Compute solution by volume approximation with gauss-points
  double d = compute_volume(*multimesh);
  cout << "Area of the total mesh (Compute_volume) "<< endl;
  cout << d << endl;

}

void assemble_MultiMeshFunction(){
  // Create meshes
  auto mesh_0 = make_shared<RectangleMesh>(Point(0.,0.), Point(2., 2.), 16, 16);
  auto mesh_1 = make_shared<RectangleMesh>(Point(1., 1.),
					   Point(3., 3.), 10, 10);
  auto mesh_2 = make_shared<RectangleMesh>(Point(-1., -1.),
					   Point(0.2, 0.2), 8, 8);

  // Build multimesh
  auto multimesh = make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  // multimesh->add(mesh_1);
  // multimesh->add(mesh_2);
  multimesh->build();

  // Create function spaces
  auto V = make_shared<MultiMeshAssemble::MultiMeshFunctionSpace>(multimesh);
  auto V0 = make_shared<MeshAssemble::FunctionSpace>(mesh_0);

  // Create forms
  MultiMeshAssemble::MultiMeshFunctional M(multimesh);
  MeshAssemble::Functional M0(mesh_0);

  // Create MultiMeshFunction
  auto v = make_shared<MultiMeshFunction>(V);
  auto v0 = make_shared<Function>(V0);

  // Set MultiMeshFunctional coefficient
  M.q = v;
  M0.q = v0;

  // Compute the MultiMeshFunctional
  double b = assemble_multimesh(M);
  cout << "MultiMeshFunctional" << endl;
  cout << b << endl;
  double b0 = assemble(M0);
  cout << "Functional" << endl;
  cout << b0 << endl;
}

void assemble_MultiMeshBilinear(){
  // Create meshes
  auto mesh_0 = make_shared<RectangleMesh>(Point(0.,0.), Point(2., 2.), 16, 16);
  auto mesh_1 = make_shared<RectangleMesh>(Point(1., 1.),
					   Point(3., 3.), 10, 10);
  auto mesh_2 = make_shared<RectangleMesh>(Point(-1., -1.),
					   Point(0.2, 0.2), 8, 8);

  // Build multimesh
  auto multimesh = make_shared<MultiMesh>();

  multimesh->add(mesh_0);
  // multimesh->add(mesh_1);
  // multimesh->add(mesh_2);
  multimesh->build();

  // Create function spaces
  auto V = make_shared<MultiMeshAssemble::MultiMeshFunctionSpace>(multimesh);
  MultiMeshAssemble::MultiMeshBilinearForm a(V,V);

  // Create MultiMeshFunction
  auto v = make_shared<MultiMeshFunction>(V);

  // Compute the MultiMeshFunctional
  double b = assemble_multimesh(a);
  cout << "MultiMeshFunctional" << endl;
  cout << b << endl;
}


int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }
  // Compute solution
  // assemble_scalar();
  assemble_MultiMeshFunction();
  assemble_MultiMeshBilinear();
  return 0;
}
