// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#include <dolfin/UnitCube.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitCube::UnitCube(uint nx, uint ny, uint nz)
{
  rename("mesh", "Mesh of the unit cube (0,1) x (0,1) x (0,1)");

  // Create nodes
  for (uint iz = 0; iz <= nz; iz++)
  {
    const real z = static_cast<real>(iz) / static_cast<real>(nz);
    for (uint iy = 0; iy <= ny; iy++)
    {
      const real y = static_cast<real>(iy) / static_cast<real>(ny);
      for (uint ix = 0; ix <= nx; ix++)
      {
	const real x = static_cast<real>(ix) / static_cast<real>(nx);
	const Point p(x, y, z);
	createNode(p);
      }
    }
  }

  // Create tetrahedra
  for (uint iz = 0; iz < nz; iz++)
  {
    for (uint iy = 0; iy < ny; iy++)
    {
      for (uint ix = 0; ix < nx; ix++)
      {
	const uint n0 = iz*(nx + 1)*(ny + 1) + iy*(nx + 1) + ix;
	const uint n1 = n0 + 1;
	const uint n2 = n0 + (nx + 1);
	const uint n3 = n1 + (nx + 1);
	const uint n4 = n0 + (nx + 1)*(ny + 1);
	const uint n5 = n1 + (nx + 1)*(ny + 1);
	const uint n6 = n2 + (nx + 1)*(ny + 1);
	const uint n7 = n3 + (nx + 1)*(ny + 1);

	createCell(n0, n1, n3, n7);
	createCell(n0, n1, n5, n7);
	createCell(n0, n5, n4, n7);
	createCell(n0, n3, n2, n7);
	createCell(n0, n6, n4, n7);
	createCell(n0, n2, n6, n7);
      }
    }
  }

  // Compute connectivity
  init();
}
//-----------------------------------------------------------------------------
