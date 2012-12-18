// Copyright (C) 2009 Anders Logg
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
// First added:  2009-02-11
// Last changed: 2012-07-05
//
// This demo program demonstrates how to extract matching sub meshes
// from a common mesh.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  cout <<"This demo is presently broken. See https://bugs.launchpad.net/dolfin/+bug/1047641" << endl;
  return 0;

  // Structure sub domain
  class Structure : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] > 1.4 - DOLFIN_EPS and x[0] < 1.6 + DOLFIN_EPS and x[1] < 0.6 + DOLFIN_EPS;
    }
  };

  // Create mesh
  RectangleMesh mesh(0.0, 0.0, 3.0, 1.0, 60, 20);

  // Create sub domain markers and mark everything as 0
  MeshFunction<std::size_t> sub_domains(mesh, mesh.topology().dim());
  sub_domains = 0;

  // Mark structure domain as 1
  Structure structure;
  structure.mark(sub_domains, 1);

  // Extract sub meshes
  SubMesh fluid_mesh(mesh, sub_domains, 0);
  SubMesh structure_mesh(mesh, sub_domains, 1);

  // Move structure mesh
  MeshGeometry& geometry = structure_mesh.geometry();
  for (VertexIterator v(structure_mesh); !v.end(); ++v)
  {
    const double* x = v->x();
    geometry.x(v->index())[0] += 0.1*x[0]*x[1];
  }

  // Move fluid mesh according to structure mesh
  fluid_mesh.move(structure_mesh);
  fluid_mesh.smooth();

  // Plot meshes
  plot(fluid_mesh);
  plot(structure_mesh);

  interactive();

  return 0;
}
