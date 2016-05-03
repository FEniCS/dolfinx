// Copyright (C) 2005-2015 Anders Logg
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
// Modified by Kristian B. Oelgaard 2009

#include <cmath>
#include <boost/multi_array.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "RectangleMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
RectangleMesh::RectangleMesh(const Point& p0, const Point& p1,
                             std::size_t nx, std::size_t ny,
                             std::string diagonal)
  : RectangleMesh(MPI_COMM_WORLD, p0, p1, nx, ny, diagonal)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
RectangleMesh::RectangleMesh(MPI_Comm comm,
                             const Point& p0, const Point& p1,
                             std::size_t nx, std::size_t ny,
                             std::string diagonal) : Mesh(comm)
{
  build(p0, p1, nx, ny, diagonal);
}
//-----------------------------------------------------------------------------
void RectangleMesh::build(const Point& p0, const Point& p1,
                          std::size_t nx, std::size_t ny,
                          std::string diagonal)
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver(this->mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }

  // Check options
  if (diagonal != "left" && diagonal != "right" && diagonal != "right/left"
          && diagonal != "left/right" && diagonal != "crossed")
  {
    dolfin_error("RectangleMesh.cpp",
                 "create rectangle",
                 "Unknown mesh diagonal definition: allowed options are \"left\", \"right\", \"left/right\", \"right/left\" and \"crossed\"");
  }

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0.x(), p1.x());
  const double x1 = std::max(p0.x(), p1.x());
  const double y0 = std::min(p0.y(), p1.y());
  const double y1 = std::max(p0.y(), p1.y());

  const double a = x0;
  const double b = x1;
  const double c = y0;
  const double d = y1;

  if (std::abs(x0 - x1) < DOLFIN_EPS || std::abs(y0 - y1) < DOLFIN_EPS)
  {
    dolfin_error("Rectangle.cpp",
                 "create rectangle",
                 "Rectangle seems to have zero width, height or depth. Consider checking your dimensions");
  }

  if (nx < 1 || ny < 1)
  {
    dolfin_error("RectangleMesh.cpp",
                 "create rectangle",
                 "Rectangle has non-positive number of vertices in some dimension: number of vertices must be at least 1 in each dimension");
  }

  rename("mesh", "Mesh of the unit square (a,b) x (c,d)");
  // Open mesh for editing
  MeshEditor editor;
  editor.open(*this, CellType::triangle, 2, 2);

  // Create vertices and cells:
  if (diagonal == "crossed")
  {
    editor.init_vertices_global((nx + 1)*(ny + 1) + nx*ny,
                                  (nx + 1)*(ny + 1) + nx*ny);
    editor.init_cells_global(4*nx*ny, 4*nx*ny);
  }
  else
  {
    editor.init_vertices_global((nx + 1)*(ny + 1), (nx + 1)*(ny + 1));
    editor.init_cells_global(2*nx*ny, 2*nx*ny);
  }

  // Storage for vertices
  std::vector<double> x(2);

  // Create main vertices:
  std::size_t vertex = 0;
  for (std::size_t iy = 0; iy <= ny; iy++)
  {
    x[1] = c + ((static_cast<double>(iy))*(d - c)/static_cast<double>(ny));
    for (std::size_t ix = 0; ix <= nx; ix++)
    {
      x[0] = a + ((static_cast<double>(ix))*(b - a)/static_cast<double>(nx));
      editor.add_vertex(vertex, x);
      vertex++;
    }
  }

  // Create midpoint vertices if the mesh type is crossed
  if (diagonal == "crossed")
  {
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      x[1] = c +(static_cast<double>(iy) + 0.5)*(d - c)/static_cast<double>(ny);
      for (std::size_t ix = 0; ix < nx; ix++)
      {
        x[0] = a + (static_cast<double>(ix) + 0.5)*(b - a)/static_cast<double>(nx);
        editor.add_vertex(vertex, x);
        vertex++;
      }
    }
  }

  // Create triangles
  std::size_t cell = 0;
  if (diagonal == "crossed")
  {
    boost::multi_array<std::size_t, 2> cells(boost::extents[4][3]);
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      for (std::size_t ix = 0; ix < nx; ix++)
      {
        const std::size_t v0 = iy*(nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        const std::size_t vmid = (nx + 1)*(ny + 1) + iy*nx + ix;

        // Note that v0 < v1 < v2 < v3 < vmid.
        cells[0][0] = v0; cells[0][1] = v1; cells[0][2] = vmid;
        cells[1][0] = v0; cells[1][1] = v2; cells[1][2] = vmid;
        cells[2][0] = v1; cells[2][1] = v3; cells[2][2] = vmid;
        cells[3][0] = v2; cells[3][1] = v3; cells[3][2] = vmid;

        // Add cells
        for (auto _cell = cells.begin(); _cell != cells.end(); ++_cell)
          editor.add_cell(cell++, *_cell);
      }
    }
  }
  else if (diagonal == "left" || diagonal == "right" || diagonal == "right/left" || diagonal == "left/right")
  {
    std::string local_diagonal = diagonal;
    boost::multi_array<std::size_t, 2> cells(boost::extents[2][3]);
    for (std::size_t iy = 0; iy < ny; iy++)
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

      for (std::size_t ix = 0; ix < nx; ix++)
      {
        const std::size_t v0 = iy*(nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        std::vector<std::size_t> cell_data;

        if(local_diagonal == "left")
        {
          cells[0][0] = v0; cells[0][1] = v1; cells[0][2] = v2;
          cells[1][0] = v1; cells[1][1] = v2; cells[1][2] = v3;
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "right";
        }
        else
        {
          cells[0][0] = v0; cells[0][1] = v1; cells[0][2] = v3;
          cells[1][0] = v0; cells[1][1] = v2; cells[1][2] = v3;
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "left";
        }
        editor.add_cell(cell++, cells[0]);
        editor.add_cell(cell++, cells[1]);
      }
    }
  }

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster(this->mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(*this);
    return;
  }
}
//-----------------------------------------------------------------------------
