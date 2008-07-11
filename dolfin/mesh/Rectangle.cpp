// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2007.
// Modified by Nuno Lopes 2008
//
// First added:  2005-12-02
// Last changed: 2008-06-19

#include "MeshEditor.h"
#include "Rectangle.h"
#include <dolfin/main/MPI.h>
#include "MPIMeshCommunicator.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Rectangle::Rectangle(real a, real b, real c, real d, uint nx, uint ny, 
                     Type type) : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::receive()) { MPIMeshCommunicator::receive(*this); return; }
  
  if ( nx < 1 || ny < 1 )
    error("Size of unit square must be at least 1 in each dimension.");

  rename("mesh", "Mesh of the unit square (a,b) x (c,d)");
  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::triangle, 2, 2);

  // Create vertices and cells:
  if (type == crisscross) 
  {
    editor.initVertices((nx+1)*(ny+1) + nx*ny);
    editor.initCells(4*nx*ny);
  } 
  else 
  {
    editor.initVertices((nx+1)*(ny+1));
    editor.initCells(2*nx*ny);
  }
  
  // Create main vertices:
  uint vertex = 0;
  for (uint iy = 0; iy <= ny; iy++) 
  {
    const real y = c + ((static_cast<real> (iy))*(d-c) / static_cast<real>(ny));
    for (uint ix = 0; ix <= nx; ix++) 
    {
      const real x = a + ((static_cast<real>(ix))*(b-a) / static_cast<real>(nx));
      editor.addVertex(vertex++, x, y);
    }
  }
  
  // Create midpoint vertices if the mesh type is crisscross
  if (type == crisscross) 
  {
    for (uint iy = 0; iy < ny; iy++) 
    {
      const real y = c +(static_cast<real>(iy) + 0.5)*(d-c)/ static_cast<real>(ny);
      for (uint ix = 0; ix < nx; ix++) 
      {
        const real x = a + (static_cast<real>(ix) + 0.5)*(b-a)/ static_cast<real>(nx);
        editor.addVertex(vertex++, x, y);
      }
    }
  }

  // Create triangles
  uint cell = 0;
  if (type == crisscross) 
  {
    for (uint iy = 0; iy < ny; iy++) 
    {
      for (uint ix = 0; ix < nx; ix++) 
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);
        const uint vmid = (nx + 1)*(ny + 1) + iy*nx + ix;
	
        // Note that v0 < v1 < v2 < v3 < vmid.
        editor.addCell(cell++, v0, v1, vmid);
        editor.addCell(cell++, v0, v2, vmid);
        editor.addCell(cell++, v1, v3, vmid);
        editor.addCell(cell++, v2, v3, vmid);
      }
    }
  } 
  else if (type == left ) 
  {
    for (uint iy = 0; iy < ny; iy++) 
    {
      for (uint ix = 0; ix < nx; ix++) 
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);

        editor.addCell(cell++, v0, v1, v2);
        editor.addCell(cell++, v1, v2, v3);
      }
    }
  } 
  else 
  { 
    for (uint iy = 0; iy < ny; iy++) 
    {
      for (uint ix = 0; ix < nx; ix++) 
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);

        editor.addCell(cell++, v0, v1, v3);
        editor.addCell(cell++, v0, v2, v3);
      }
    }
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::broadcast()) { MPIMeshCommunicator::broadcast(*this); }
}
