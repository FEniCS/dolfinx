// Copyright (C) 2008-2009 Solveig Bruvoll and Anders Logg
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
// First added:  2008-05-02
// Last changed: 2012-07-05
//
// This demo demonstrates how to move the vertex coordinates
// of a boundary mesh and then updating the interior vertex
// coordinates of the original mesh by suitably interpolating
// the vertex coordinates (useful for implementation of ALE
// methods).

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Create mesh
  auto mesh = std::make_shared<UnitSquareMesh>(20, 20);

  // Create boundary mesh
  BoundaryMesh boundary(*mesh, "exterior");

  // Move vertices in boundary
  MeshGeometry& geometry = boundary.geometry();

  std::vector<double> x(2);
  for (VertexIterator v(boundary); !v.end(); ++v)
  {
    std::size_t i = v->index();
    x[0] = geometry.x(i, 0)*3.0;
    x[1] = geometry.x(i, 1) + 0.1*sin(5.0*x[0]);
    geometry.set(i, x.data());
  }

  // Move mesh
  ALE::move(mesh, boundary);

  // Plot mesh
  plot(*mesh);
  interactive();

  // Write mesh to file
  File file("mesh_deformed.pvd");
  file << *mesh;

  return 0;
}
