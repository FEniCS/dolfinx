// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-02
// Last changed: 2006-08-07

#include <dolfin/MeshEditor.h>
#include <dolfin/UnitSquare.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitSquare::UnitSquare(uint nx, uint ny) : Mesh()
{
  rename("mesh", "Mesh of the unit square (0,1) x (0,1)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::triangle, 2, 2);

  // Create vertices
  editor.initVertices((nx+1)*(ny+1));
  uint vertex = 0;
  for (uint iy = 0; iy <= ny; iy++)
  {
    const real y = static_cast<real>(iy) / static_cast<real>(ny);
    for (uint ix = 0; ix <= nx; ix++)
    {
      const real x = static_cast<real>(ix) / static_cast<real>(nx);
      editor.addVertex(vertex++, x, y);
    }
  }
  
  // Create triangles
  editor.initCells(2*nx*ny);
  uint cell = 0;
  for (uint iy = 0; iy < ny; iy++)
  {
    for (uint ix = 0; ix < nx; ix++)
    {
      const uint v0 = iy*(nx + 1) + ix;
      const uint v1 = v0 + 1;
      const uint v2 = v0 + (nx + 1);
      const uint v3 = v1 + (nx + 1);

      editor.addCell(cell++, v0, v1, v3);
      editor.addCell(cell++, v0, v3, v2);
    }
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
