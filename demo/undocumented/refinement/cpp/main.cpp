// Copyright (C) 2006-2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman, 2006.
//
// First added:  2006-10-26
// Last changed: 2011-02-04

#include <dolfin.h>

using namespace dolfin;

int main()
{
  File file("mesh.pvd");

  // Create mesh of unit square
  UnitSquare unit_square(5, 5);
  Mesh mesh(unit_square);
  file << mesh;

  // Uniform refinement
  mesh = refine(mesh);
  file << mesh;

  // Refine mesh close to x = (0.5, 0.5)
  Point p(0.5, 0.5);
  for (unsigned int i = 0; i < 5; i++)
  {
    // Mark cells for refinement
    MeshFunction<bool> cell_markers(mesh, mesh.topology().dim(), false);
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      if (c->midpoint().distance(p) < 0.1)
        cell_markers[*c] = true;
    }

    // Refine mesh
    mesh = refine(mesh, cell_markers);

    file << mesh;
    plot(mesh);
  }

  return 0;
}
