// Copyright (C) 2005-2009 Anders Logg
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
// Modified by Garth N. Wells 2007.
// Modified by Kristian B. Oelgaard 2009.
//
// First added:  2005-12-02
// Last changed: 2009-09-29

#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitSquare.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitSquare::UnitSquare(uint nx, uint ny, std::string diagonal) : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver())
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }

  if (diagonal != "left" && diagonal != "right" && diagonal != "right/left"
      && diagonal != "left/right"  && diagonal != "crossed")
    dolfin_error("UnitSquare.cpp",
                 "create unit square",
                 "Unknown mesh diagonal definition: allowed options are \"left\", \"right\", \"left/right\", \"right/left\" and \"crossed\"");

  if (nx < 1 || ny < 1)
    dolfin_error("UnitSquare.cpp",
                 "create unit square",
                 "Unit square has non-positive number of vertices in some dimension: number of vertices must be at least 1 in each dimension");

  rename("mesh", "Mesh of the unit square (0,1) x (0,1)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::triangle, 2, 2);

  // Create vertices and cells:
  if (diagonal == "crossed")
  {
    editor.init_vertices((nx + 1)*(ny + 1) + nx*ny);
    editor.init_cells(4*nx*ny);
  }
  else
  {
    editor.init_vertices((nx + 1)*(ny + 1));
    editor.init_cells(2*nx*ny);
  }

  // Storage for vertex coordinates
  std::vector<double> x(2);

  // Create main vertices:
  uint vertex = 0;
  for (uint iy = 0; iy <= ny; iy++)
  {
    x[1] = static_cast<double>(iy)/static_cast<double>(ny);
    for (uint ix = 0; ix <= nx; ix++)
    {
      x[0] = static_cast<double>(ix)/static_cast<double>(nx);
      editor.add_vertex(vertex++, x);
    }
  }

  // Create midpoint vertices if the mesh type is crisscross
  if (diagonal == "crossed")
  {
    for (uint iy = 0; iy < ny; iy++)
    {
      x[1] = (static_cast<double>(iy) + 0.5) / static_cast<double>(ny);
      for (uint ix = 0; ix < nx; ix++)
      {
        x[0] = (static_cast<double>(ix) + 0.5) / static_cast<double>(nx);
        editor.add_vertex(vertex++, x);
      }
    }
  }

  // Create triangles
  uint cell = 0;
  if (diagonal == "crossed")
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
        editor.add_cell(cell++, v0, v1, vmid);
        editor.add_cell(cell++, v0, v2, vmid);
        editor.add_cell(cell++, v1, v3, vmid);
        editor.add_cell(cell++, v2, v3, vmid);
      }
    }
  }
  else if (diagonal == "left" || diagonal == "right" || diagonal == "right/left" || diagonal == "left/right")
  {
    std::string local_diagonal = diagonal;
    for (uint iy = 0; iy < ny; iy++)
    {
      // Set up alternating diagonal
      if (diagonal == "right/left")
      {
        if (iy % 2)
          local_diagonal = "right";
        else
          local_diagonal = "left";
      }
      if (diagonal == "left/right")
      {
        if (iy % 2)
          local_diagonal = "left";
        else
          local_diagonal = "right";
      }

      for (uint ix = 0; ix < nx; ix++)
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);

        if(local_diagonal == "left")
        {
          editor.add_cell(cell++, v0, v1, v2);
          editor.add_cell(cell++, v1, v2, v3);
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "right";
        }
        else
        {
          editor.add_cell(cell++, v0, v1, v3);
          editor.add_cell(cell++, v0, v2, v3);
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "left";
        }
      }
    }
  }
  else
    dolfin_error("UnitSquare.cpp",
                 "create unit square",
                 "Unknown mesh diagonal definition: allowed options are \"left\", \"right\", \"left/right\", \"right/left\" and \"crossed\"");

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster())
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }
}
//-----------------------------------------------------------------------------
