// Copyright (C) 2005-2008 Anders Logg
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
// Modified by Garth N. Wells, 2007.
// Modified by Nuno Lopes, 2008.
//
// First added:  2005-12-02
// Last changed: 2008-11-13

#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include "MeshPartitioning.h"
#include "MeshEditor.h"
#include "Box.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Box::Box(double x0, double y0, double z0,
         double x1, double y1, double z1,
         uint nx, uint ny, uint nz) : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver()) { MeshPartitioning::build_distributed_mesh(*this); return; }

  const double a = x0;
  const double b = x1;
  const double c = y0;
  const double d = y1;
  const double e = z0;
  const double f = z1;

  if (std::abs(x0 - x1) < DOLFIN_EPS || std::abs(y0 - y1) < DOLFIN_EPS || std::abs(z0 - z1) < DOLFIN_EPS )
    error("Box must have nonzero width, height and depth.");

  if ( nx < 1 || ny < 1 || nz < 1 )
    error("Size of box must be at least 1 in each dimension.");

  rename("mesh", "Mesh of the cuboid (a,b) x (c,d) x (e,f)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::tetrahedron, 3, 3);

  // Create vertices
  editor.init_vertices((nx+1)*(ny+1)*(nz+1));
  uint vertex = 0;
  for (uint iz = 0; iz <= nz; iz++)
  {
    const double z = e + (static_cast<double>(iz))*(f-e) / static_cast<double>(nz);
    for (uint iy = 0; iy <= ny; iy++)
    {
      const double y = c + (static_cast<double>(iy))*(d-c) / static_cast<double>(ny);
      for (uint ix = 0; ix <= nx; ix++)
      {
        const double x = a + (static_cast<double>(ix))*(b-a) / static_cast<double>(nx);
        editor.add_vertex(vertex++, x, y, z);
      }
    }
  }

  // Create tetrahedra
  editor.init_cells(6*nx*ny*nz);
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

        editor.add_cell(cell++, v0, v1, v3, v7);
        editor.add_cell(cell++, v0, v1, v7, v5);
        editor.add_cell(cell++, v0, v5, v7, v4);
        editor.add_cell(cell++, v0, v3, v2, v7);
        editor.add_cell(cell++, v0, v6, v4, v7);
        editor.add_cell(cell++, v0, v2, v6, v7);
      }
    }
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster()) { MeshPartitioning::build_distributed_mesh(*this); return; }
}
//-----------------------------------------------------------------------------
