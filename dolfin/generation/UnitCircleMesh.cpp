// Copyright (C) 2005-2012 Anders Logg
// AL: I don't think I wrote this file, who did?
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
// Modified by Garth N. Wells 2007
// Modified by Nuno Lopes 2008
// Modified by Anders Logg 2012
//
// First added:  2005-12-02
// Last changed: 2012-03-06

#include <boost/assign.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitCircleMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
UnitCircleMesh::UnitCircleMesh(uint n, std::string diagonal,
                               std::string transformation) : Mesh()
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver())
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }

  if (diagonal != "left" && diagonal != "right" && diagonal != "crossed")
  {
    dolfin_error("UnitCircleMesh.cpp",
                 "create unit circle",
                 "Unknown mesh diagonal definition: Allowed options are \"left\", \"right\" and \"crossed\"");
  }

  if (transformation != "maxn" && transformation != "sumn" && transformation != "rotsumn")
  {
    dolfin_error("UnitCircleMesh.cpp",
                 "create unit circle",
                 "Unknown transformation '%s' in UnitCircleMesh. Allowed options are \"maxn\", \"sumn\" and \"rotsumn\"",
                 transformation.c_str());
  }

  if (n < 1)
  {
    dolfin_error("UnitCircleMesh.cpp",
                 "create unit circle",
                 "Size of unit square must be at least 1");
  }

  const uint nx = n;
  const uint ny = n;

  rename("mesh", "Mesh of the unit circle");

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

  // Data structure for creating vertices
  std::vector<double> x(2);

  // Create main vertices
  uint vertex = 0;
  for (uint iy = 0; iy <= ny; iy++)
  {
    x[1] = -1.0 + static_cast<double>(iy)*2.0/static_cast<double>(ny);
    for (uint ix = 0; ix <= nx; ix++)
    {
      x[0] = -1.0 + static_cast<double>(ix)*2.0 / static_cast<double>(nx);
      const std::vector<double> x_trans = transform(x, transformation);
      editor.add_vertex(vertex, x_trans);
      vertex++;
    }
  }

  // Create midpoint vertices if the mesh type is crisscross
  if (diagonal == "crossed")
  {
    for (uint iy = 0; iy < ny; iy++)
    {
      x[1] = -1.0 + (static_cast<double>(iy) + 0.5)*2.0 / static_cast<double>(ny);
      for (uint ix = 0; ix < nx; ix++)
      {
        x[0] = -1.0 + (static_cast<double>(ix) + 0.5)*2.0 / static_cast<double>(nx);
        const std::vector<double> x_trans = transform(x, transformation);
        editor.add_vertex(vertex, x_trans);
        vertex++;
      }
    }
  }

  // Create triangles
  uint cell = 0;
  if (diagonal == "crossed")
  {
    std::vector<std::vector<uint> > cells(4, std::vector<uint>(3));
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
        cells[0][0] = v0; cells[0][1] = v1; cells[0][2] = vmid;
        cells[1][0] = v0; cells[1][1] = v2; cells[1][2] = vmid;
        cells[2][0] = v1; cells[2][1] = v3; cells[2][2] = vmid;
        cells[3][0] = v2; cells[3][1] = v3; cells[3][2] = vmid;

        // Add cells
        std::vector<std::vector<uint> >::const_iterator _cell;
        for (_cell = cells.begin(); _cell != cells.end(); ++_cell)
          editor.add_cell(cell++, *_cell);

      }
    }
  }
  else if (diagonal == "left" ||  diagonal == "right")
  {
    std::vector<std::vector<uint> > cells(2, std::vector<uint>(3));
    for (uint iy = 0; iy < ny; iy++)
    {
      for (uint ix = 0; ix < nx; ix++)
      {
        const uint v0 = iy*(nx + 1) + ix;
        const uint v1 = v0 + 1;
        const uint v2 = v0 + (nx + 1);
        const uint v3 = v1 + (nx + 1);
        std::vector<uint> cell_data;

        if(diagonal == "left")
        {
          cells[0][0] = v0; cells[0][1] = v1; cells[0][2] = v2;
          cells[1][0] = v1; cells[1][1] = v2; cells[1][2] = v3;
        }
        else
        {
          cells[0][0] = v0; cells[0][1] = v1; cells[0][2] = v3;
          cells[1][0] = v0; cells[1][1] = v2; cells[1][2] = v3;
        }
        editor.add_cell(cell++, cells[0]);
        editor.add_cell(cell++, cells[1]);
      }
    }
  }
  else
  {
    dolfin_error("UnitCircleMesh.cpp",
                 "create unit circle",
                 "Unknown mesh diagonal definition: Allowed options are \"left\", \"right\" and \"crossed\"");
  }

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
std::vector<double> UnitCircleMesh::transform(const std::vector<double>& x,
                                              const std::string transformation)
{
  if (std::abs(x[0]) < DOLFIN_EPS && std::abs(x[1]) < DOLFIN_EPS)
    return x;

  std::vector<double> x_trans(2);
  const double dist = sqrt(x[0]*x[0] + x[1]*x[1]);
  if(transformation == "maxn")
  {
    x_trans[0] = x[0]*max(x)/dist;
    x_trans[1] = x[1]*max(x)/dist;
  }
  else if (transformation == "sumn")
  {
    // FIXME: This option should either be removed or fixed
    dolfin_error("UnitCircleMesh.cpp",
                 "transform to unit circle",
                 "'sumn' mapping for a UnitCircleMesh is broken");
    x_trans[0] = x[0]*(std::abs(x[0]) + std::abs(x[1]))/dist;
    x_trans[1] = x[1]*(std::abs(x[0]) + std::abs(x[1]))/dist;
  }
  else if (transformation == "rotsumn")
  {
    const double xx = 0.5*(x[0] + x[1]);
    const double yy = 0.5*(-x[0] + x[1]);
    x_trans[0] = xx*(std::abs(xx) + std::abs(yy))/sqrt(xx*xx+yy*yy);
    x_trans[1] = yy*(std::abs(xx) + std::abs(yy))/sqrt(xx*xx+yy*yy);
  }
  else
  {
    dolfin_error("UnitCircleMesh.cpp",
                 "transform to unit circle",
                 "Unknown transformation '%s' in UnitCircleMesh. Allowed options are \"maxn\", \"sumn\" and \"rotsumn\"",
                 transformation.c_str());
  }

  return x_trans;
}
//-----------------------------------------------------------------------------
