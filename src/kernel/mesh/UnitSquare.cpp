// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005-12-01

#include <dolfin/UnitSquare.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitSquare::UnitSquare(uint nx, uint ny) : Mesh()
{
  rename("mesh", "Mesh of the unit square (0,1) x (0,1)");

  // Create vertices
  for (uint iy = 0; iy <= ny; iy++)
  {
    const real y = static_cast<real>(iy) / static_cast<real>(ny);
    for (uint ix = 0; ix <= nx; ix++)
    {
      const real x = static_cast<real>(ix) / static_cast<real>(nx);
      const Point p(x, y);
      createVertex(p);
    }
  }
  
  // Create triangles
  for (uint iy = 0; iy < ny; iy++)
  {
    for (uint ix = 0; ix < nx; ix++)
    {
      const uint n0 = iy*(nx + 1) + ix;
      const uint n1 = n0 + 1;
      const uint n2 = n0 + (nx + 1);
      const uint n3 = n1 + (nx + 1);

      createCell(n0, n1, n3);
      createCell(n0, n3, n2);
    }
  }

  // Compute connectivity
  init();
}
//-----------------------------------------------------------------------------
