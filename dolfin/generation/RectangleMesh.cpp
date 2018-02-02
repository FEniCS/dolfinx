// Copyright (C) 2005-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "RectangleMesh.h"
#include <Eigen/Dense>
#include <cmath>
#include <dolfin/common/MPI.h>
#include <dolfin/common/constants.h>
#include <dolfin/mesh/MeshPartitioning.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Mesh RectangleMesh::build_tri(MPI_Comm comm, const std::array<Point, 2>& p,
                              std::array<std::size_t, 2> n,
                              std::string diagonal)
{
  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom(
        0, 2);
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> topo(0, 3);
    Mesh mesh(comm, CellType::Type::triangle, geom, topo);
    mesh.order();
    MeshPartitioning::build_distributed_mesh(mesh);
    return mesh;
  }

  // Check options
  if (diagonal != "left" && diagonal != "right" && diagonal != "right/left"
      && diagonal != "left/right" && diagonal != "crossed")
  {
    dolfin_error("RectangleMesh.cpp", "create rectangle",
                 "Unknown mesh diagonal definition: allowed options are "
                 "\"left\", \"right\", \"left/right\", \"right/left\" and "
                 "\"crossed\"");
  }

  const Point& p0 = p[0];
  const Point& p1 = p[1];

  const std::size_t nx = n[0];
  const std::size_t ny = n[1];

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0.x(), p1.x());
  const double x1 = std::max(p0.x(), p1.x());
  const double y0 = std::min(p0.y(), p1.y());
  const double y1 = std::max(p0.y(), p1.y());

  const double a = x0;
  const double b = x1;
  const double ab = (b - a) / static_cast<double>(nx);
  const double c = y0;
  const double d = y1;
  const double cd = (d - c) / static_cast<double>(ny);

  if (std::abs(x0 - x1) < DOLFIN_EPS || std::abs(y0 - y1) < DOLFIN_EPS)
  {
    dolfin_error("Rectangle.cpp", "create rectangle",
                 "Rectangle seems to have zero width, height or depth. "
                 "Consider checking your dimensions");
  }

  if (nx < 1 || ny < 1)
  {
    dolfin_error("RectangleMesh.cpp", "create rectangle",
                 "Rectangle has non-positive number of vertices in some "
                 "dimension: number of vertices must be at least 1 in each "
                 "dimension");
  }

  // Create vertices and cells
  std::size_t nv, nc;
  if (diagonal == "crossed")
  {
    nv = (nx + 1) * (ny + 1) + nx * ny;
    nc = 4 * nx * ny;
  }
  else
  {
    nv = (nx + 1) * (ny + 1);
    nc = 2 * nx * ny;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom(
      nv, 2);
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> topo(nc,
                                                                           3);

  // Create main vertices
  std::size_t vertex = 0;
  for (std::size_t iy = 0; iy <= ny; iy++)
  {
    const double x1 = c + cd * static_cast<double>(iy);
    for (std::size_t ix = 0; ix <= nx; ix++)
    {
      geom(vertex, 0) = a + ab * static_cast<double>(ix);
      geom(vertex, 1) = x1;
      ++vertex;
    }
  }

  // Create midpoint vertices if the mesh type is crossed
  if (diagonal == "crossed")
  {
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      const double x1 = c + cd * (static_cast<double>(iy) + 0.5);
      for (std::size_t ix = 0; ix < nx; ix++)
      {
        geom(vertex, 0) = a + ab * (static_cast<double>(ix) + 0.5);
        geom(vertex, 1) = x1;
        ++vertex;
      }
    }
  }

  // Create triangles
  std::size_t cell = 0;
  if (diagonal == "crossed")
  {
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      for (std::size_t ix = 0; ix < nx; ix++)
      {
        const std::size_t v0 = iy * (nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        const std::size_t vmid = (nx + 1) * (ny + 1) + iy * nx + ix;

        // Note that v0 < v1 < v2 < v3 < vmid.
        topo.row(cell) << v0, v1, vmid;
        ++cell;
        topo.row(cell) << v0, v2, vmid;
        ++cell;
        topo.row(cell) << v1, v3, vmid;
        ++cell;
        topo.row(cell) << v2, v3, vmid;
        ++cell;
      }
    }
  }
  else if (diagonal == "left" || diagonal == "right" || diagonal == "right/left"
           || diagonal == "left/right")
  {
    std::string local_diagonal = diagonal;
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
        const std::size_t v0 = iy * (nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        std::vector<std::size_t> cell_data;

        if (local_diagonal == "left")
        {
          topo.row(cell) << v0, v1, v2;
          ++cell;
          topo.row(cell) << v1, v2, v3;
          ++cell;
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "right";
        }
        else
        {
          topo.row(cell) << v0, v1, v3;
          ++cell;
          topo.row(cell) << v0, v2, v3;
          ++cell;
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "left";
        }
      }
    }
  }

  Mesh mesh(comm, CellType::Type::triangle, geom, topo);
  mesh.order();

  MeshPartitioning::build_distributed_mesh(mesh);
  return mesh;
}
//-----------------------------------------------------------------------------
Mesh RectangleMesh::build_quad(MPI_Comm comm, const std::array<Point, 2>& p,
                               std::array<std::size_t, 2> n)
{
  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom(0, 2);
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> topo(0, 4);
    Mesh mesh(comm, CellType::Type::quadrilateral, geom, topo);
    MeshPartitioning::build_distributed_mesh(mesh);
    return mesh;
  }

  const std::size_t nx = n[0];
  const std::size_t ny = n[1];

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom(
      (nx + 1) * (ny + 1), 2);
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> topo(
      nx * ny, 4);

  const double a = p[0][0];
  const double b = p[1][0];
  const double ab = (b - a) / static_cast<double>(nx);

  const double c = p[0][1];
  const double d = p[1][1];
  const double cd = (d - c) / static_cast<double>(ny);

  // Create vertices
  std::size_t vertex = 0;
  for (std::size_t iy = 0; iy <= ny; iy++)
  {
    double x1 = c + cd * static_cast<double>(iy);
    for (std::size_t ix = 0; ix <= nx; ix++)
    {
      geom(vertex, 0) = a + ab * static_cast<double>(ix);
      geom(vertex, 1) = x1;
      ++vertex;
    }
  }

  // Create rectangles
  std::size_t cell = 0;
  for (std::size_t iy = 0; iy < ny; iy++)
    for (std::size_t ix = 0; ix < nx; ix++)
    {
      const std::size_t i0 = iy * (nx + 1);
      topo(cell, 0) = i0 + ix;
      topo(cell, 1) = i0 + ix + 1;
      topo(cell, 2) = i0 + ix + nx + 1;
      topo(cell, 3) = i0 + ix + nx + 2;
      ++cell;
    }

  Mesh mesh(comm, CellType::Type::quadrilateral, geom, topo);
  MeshPartitioning::build_distributed_mesh(mesh);
  return mesh;
}
//-----------------------------------------------------------------------------
