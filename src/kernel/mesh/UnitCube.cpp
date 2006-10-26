// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-02
// Last changed: 2006-08-07

#include <dolfin/MeshEditor.h>
#include <dolfin/UnitCube.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitCube::UnitCube(uint nx, uint ny, uint nz) : Mesh()
{
  rename("mesh", "Mesh of the unit cube (0,1) x (0,1) x (0,1)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::tetrahedron, 3, 3);

  // Create vertices
  editor.initVertices((nx+1)*(ny+1)*(nz+1));
  uint vertex = 0;
  for (uint iz = 0; iz <= nz; iz++)
  {
    const real z = static_cast<real>(iz) / static_cast<real>(nz);
    for (uint iy = 0; iy <= ny; iy++)
    {
      const real y = static_cast<real>(iy) / static_cast<real>(ny);
      for (uint ix = 0; ix <= nx; ix++)
      {
	const real x = static_cast<real>(ix) / static_cast<real>(nx);
	editor.addVertex(vertex++, x, y, z);
      }
    }
  }

  // Create tetrahedra
  editor.initCells(6*nx*ny*nz);
  uint cell = 0;
  for (uint iz = 0; iz < nz; iz++)
  {
    for (uint iy = 0; iy < ny; iy++)
    {
      for (uint ix = 0; ix < nx; ix++)
      {
	const uint v0 = iz*(nx + 1)*(ny + 1) + iy*(nx + 1) + ix;
	const uint v1 = v0 + 1;
	const uint v2 = v0 + (nx + 1);
	const uint v3 = v1 + (nx + 1);
	const uint v4 = v0 + (nx + 1)*(ny + 1);
	const uint v5 = v1 + (nx + 1)*(ny + 1);
	const uint v6 = v2 + (nx + 1)*(ny + 1);
	const uint v7 = v3 + (nx + 1)*(ny + 1);

	editor.addCell(cell++, v0, v1, v3, v7);
	editor.addCell(cell++, v0, v1, v7, v5);
	editor.addCell(cell++, v0, v5, v7, v4);
	editor.addCell(cell++, v0, v3, v2, v7);
	editor.addCell(cell++, v0, v6, v4, v7);
	editor.addCell(cell++, v0, v2, v6, v7);
      }
    }
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
