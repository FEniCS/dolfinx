// Copyright (C) 2006-2011 Anders Logg
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
// Modified by Johan Hoffman, 2006.
//
// First added:  2006-10-26
// Last changed: 2012-11-12

#include <dolfin.h>

using namespace dolfin;

int main()
{
  File file("mesh.pvd");

  // Create mesh of unit square
  auto mesh = std::make_shared<Mesh>(UnitSquareMesh(5, 5));

  file << *mesh;

  // Uniform refinement
  mesh = std::make_shared<Mesh>(refine(*mesh));
  file << *mesh;

  // Refine mesh close to x = (0.5, 0.5)
  Point p(0.5, 0.5);
  for (unsigned int i = 0; i < 5; i++)
  {
    // Mark cells for refinement
    MeshFunction<bool> cell_markers(mesh, mesh->topology().dim(), false);
    for (CellIterator c(*mesh); !c.end(); ++c)
    {
      if (c->midpoint().distance(p) < 0.1)
        cell_markers[*c] = true;
    }

    // Refine mesh
    mesh = std::make_shared<Mesh>(refine(*mesh, cell_markers));

    file << *mesh;
    plot(*mesh);
    interactive();
  }

  return 0;
}
