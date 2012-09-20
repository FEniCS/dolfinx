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
//
// First added:  2005-12-02
// Last changed: 2010-10-19

#include <boost/assign.hpp>

#include <dolfin/common/timing.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitCube.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitCube::UnitCube(uint nx, uint ny, uint nz) : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver())
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }

  // Check input
  if ( nx < 1 || ny < 1 || nz < 1 )
  {
    dolfin_error("UnitCube.cpp",
                 "create unit cube",
                 "Cube has non-positive number of vertices in some dimension: number of vertices must be at least 1 in each dimension");
  }

  // Set name
  rename("mesh", "Mesh of the unit cube (0,1) x (0,1) x (0,1)");

  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::tetrahedron, 3, 3);

  // Storage for vertex coordinates
  std::vector<double> x(3);

  // Create vertices
  editor.init_vertices((nx+1)*(ny+1)*(nz+1));
  uint vertex = 0;
  for (uint iz = 0; iz <= nz; iz++)
  {
    x[2] = static_cast<double>(iz) / static_cast<double>(nz);
    for (uint iy = 0; iy <= ny; iy++)
    {
      x[1] = static_cast<double>(iy) / static_cast<double>(ny);
      for (uint ix = 0; ix <= nx; ix++)
      {
        x[0] = static_cast<double>(ix) / static_cast<double>(nx);
        editor.add_vertex(vertex, vertex, x);
        vertex++;
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

        // Data structure to hold cells
        std::vector<std::vector<uint> > cells;

        // Note that v0 < v1 < v2 < v3 < vmid.
        cells.push_back(boost::assign::list_of(v0)(v1)(v3)(v7));
        cells.push_back(boost::assign::list_of(v0)(v1)(v7)(v5));
        cells.push_back(boost::assign::list_of(v0)(v5)(v7)(v4));
        cells.push_back(boost::assign::list_of(v0)(v3)(v2)(v7));
        cells.push_back(boost::assign::list_of(v0)(v6)(v4)(v7));
        cells.push_back(boost::assign::list_of(v0)(v2)(v6)(v7));

        // Add cells
        std::vector<std::vector<uint> >::const_iterator _cell;
        for (_cell = cells.begin(); _cell != cells.end(); ++_cell)
          editor.add_cell(cell++, *_cell);
      }
    }
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster())
    MeshPartitioning::build_distributed_mesh(*this);
}
//-----------------------------------------------------------------------------
