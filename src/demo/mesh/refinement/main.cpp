// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-10-26
// Last changed: 2007-05-30

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Create mesh of unit square
  UnitSquare mesh(5, 5);
  plot(mesh);

  // Uniform refinement
  mesh.refine();
  plot(mesh);

  // Refine mesh at x = t
  real t = 0.0;
  while (t < 1.0)
  {
    {
      // Mark cells for refinement
      MeshFunction<bool> cell_markers(mesh, mesh.topology().dim());
      cell_markers = false;
      for (CellIterator c(mesh); !c.end(); ++c)
      {
        if (std::abs(c->midpoint().x() - t) < 0.1 && c->diameter() > 0.11)
          cell_markers(*c) = true;
      }

      // Refine mesh
      mesh.refine(cell_markers);
      //mesh.smooth();
      plot(mesh);
    }

    {
      // Mark cell for coarsening
      MeshFunction<bool> cell_markers(mesh, mesh.topology().dim());
      cell_markers = false;
      for (CellIterator c(mesh); !c.end(); ++c)
      {
        if (std::abs(c->midpoint().x() - t) < 0.1 && c->diameter() < 0.11)
          cell_markers(*c) = true;
      }

      // Coarsen mesh
      mesh.coarsen(cell_markers);
      //mesh.smooth();
      plot(mesh);
    }

    t += 0.1;
  }

  return 0;
}
