// Copyright (C) 2005-2015 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "RectangleMesh.h"
#include <cfloat>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <dolfinx/mesh/utils.h>


using namespace dolfinx;
using namespace dolfinx::generation;

namespace
{
//-----------------------------------------------------------------------------
mesh::Mesh build_tri(MPI_Comm comm,
                     const std::array<std::array<double, 3>, 2>& p,
                     std::array<std::size_t, 2> n,
                     const mesh::GhostMode ghost_mode,
                     const mesh::CellPartitionFunction& partitioner,
                     const std::string& diagonal)
{
  fem::CoordinateElement element(mesh::CellType::triangle, 1);

  // Receive mesh if not rank 0
  if (dolfinx::MPI::rank(comm) != 0)
  {
    xt::xtensor<double, 2> geom({0, 2});
    xt::xtensor<std::int64_t, 2> cells({0, 3});
    auto [data, offset] = graph::create_adjacency_data(cells);
    return mesh::create_mesh(
        comm,
        graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
        element, geom, ghost_mode, partitioner);
  }

  // Check options
  if (diagonal != "left" && diagonal != "right" && diagonal != "right/left"
      && diagonal != "left/right" && diagonal != "crossed")
  {
    throw std::runtime_error("Unknown mesh diagonal definition.");
  }

  const std::array<double, 3> p0 = p[0];
  const std::array<double, 3> p1 = p[1];

  const std::size_t nx = n[0];
  const std::size_t ny = n[1];

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0[0], p1[0]);
  const double x1 = std::max(p0[0], p1[0]);
  const double y0 = std::min(p0[1], p1[1]);
  const double y1 = std::max(p0[1], p1[1]);

  const double a = x0;
  const double b = x1;
  const double ab = (b - a) / static_cast<double>(nx);
  const double c = y0;
  const double d = y1;
  const double cd = (d - c) / static_cast<double>(ny);

  if (std::abs(x0 - x1) < DBL_EPSILON || std::abs(y0 - y1) < DBL_EPSILON)
  {
    throw std::runtime_error("Rectangle seems to have zero width, height or "
                             "depth. Check dimensions");
  }

  if (nx < 1 || ny < 1)
  {
    throw std::runtime_error(
        "Rectangle has non-positive number of vertices in some dimension: "
        "number of vertices must be at least 1 in each dimension");
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

  xt::xtensor<double, 2> geom({nv, 2});
  xt::xtensor<std::int64_t, 2> cells({nc, 3});

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

        // Note that v0 < v1 < v2 < v3 < vmid
        xt::xtensor_fixed<std::size_t, xt::xshape<4, 3>> c
            = {{v0, v1, vmid}, {v0, v2, vmid}, {v1, v3, vmid}, {v2, v3, vmid}};
        std::size_t offset = iy * nx + ix;

        // Note: we would like to assign to a view, but this does not
        // work correctly with the Intel icpx compiler
        // xt::view(cells, xt::range(4 * offset, 4 * offset + 4), xt::all()) =
        // c;
        auto _cell
            = xt::view(cells, xt::range(4 * offset, 4 * offset + 4), xt::all());
        _cell.assign(c);
      }
    }
  }
  else if (diagonal == "left" or diagonal == "right" or diagonal == "right/left"
           or diagonal == "left/right")
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

        std::size_t offset = iy * nx + ix;
        if (local_diagonal == "left")
        {
          xt::xtensor_fixed<std::size_t, xt::xshape<2, 3>> c
              = {{v0, v1, v2}, {v1, v2, v3}};
          // Note: we would like to assign to a view, but this does not
          // work correctly with the Intel icpx compiler
          // xt::view(cells, xt::range(2 * offset, 2 * offset + 2), xt::all()) =
          // c;
          auto _cell = xt::view(cells, xt::range(2 * offset, 2 * offset + 2),
                                xt::all());
          _cell.assign(c);
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "right";
        }
        else
        {
          xt::xtensor_fixed<std::size_t, xt::xshape<2, 3>> c
              = {{v0, v1, v3}, {v0, v2, v3}};
          // Note: we would like to assign to a view, but this does not
          // work correctly with the Intel icpx compiler
          // xt::view(cells, xt::range(2 * offset, 2 * offset + 2), xt::all()) =
          // c;
          auto _cell = xt::view(cells, xt::range(2 * offset, 2 * offset + 2),
                                xt::all());
          _cell.assign(c);
          if (diagonal == "right/left" || diagonal == "left/right")
            local_diagonal = "left";
        }
      }
    }
  }

  auto [data, offset] = graph::create_adjacency_data(cells);
  return mesh::create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}

} // namespace
//-----------------------------------------------------------------------------
mesh::Mesh build_quad(MPI_Comm comm,
                      const std::array<std::array<double, 3>, 2> p,
                      std::array<std::size_t, 2> n,
                      const mesh::GhostMode ghost_mode,
                      const mesh::CellPartitionFunction& partitioner)
{
  fem::CoordinateElement element(mesh::CellType::quadrilateral, 1);

  // Receive mesh if not rank 0
  if (dolfinx::MPI::rank(comm) != 0)
  {
    xt::xtensor<double, 2> geom({0, 2});
    xt::xtensor<std::int64_t, 2> cells({0, 4});
    auto [data, offset] = graph::create_adjacency_data(cells);
    return mesh::create_mesh(
        comm,
        graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
        element, geom, ghost_mode, partitioner);
  }

  const std::size_t nx = n[0];
  const std::size_t ny = n[1];

  const double a = p[0][0];
  const double b = p[1][0];
  const double ab = (b - a) / static_cast<double>(nx);

  const double c = p[0][1];
  const double d = p[1][1];
  const double cd = (d - c) / static_cast<double>(ny);

  // Create vertices
  xt::xtensor<double, 2> geom({(nx + 1) * (ny + 1), 2});
  std::size_t vertex = 0;
  for (std::size_t ix = 0; ix <= nx; ix++)
  {
    double x0 = a + ab * static_cast<double>(ix);
    for (std::size_t iy = 0; iy <= ny; iy++)
    {
      geom(vertex, 0) = x0;
      geom(vertex, 1) = c + cd * static_cast<double>(iy);
      ++vertex;
    }
  }

  // Create rectangles
  xt::xtensor<std::int64_t, 2> cells({nx * ny, 4});
  for (std::size_t ix = 0; ix < nx; ix++)
  {
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      const std::size_t i0 = ix * (ny + 1);
      std::size_t cell = ix * ny + iy;
      xt::xtensor_fixed<std::size_t, xt::xshape<4>> c
          = {i0 + iy, i0 + iy + 1, i0 + iy + ny + 1, i0 + iy + ny + 2};
      // Note: we would like to assign to a view, but this does not
      // work correctly with the Intel icpx compiler
      // xt::row(cells, cell) = c;
      auto _cell = xt::row(cells, cell);
      _cell.assign(c);
    }
  }

  auto [data, offset] = graph::create_adjacency_data(cells);
  return mesh::create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------
mesh::Mesh RectangleMesh::create(MPI_Comm comm,
                                 const std::array<std::array<double, 3>, 2>& p,
                                 std::array<std::size_t, 2> n,
                                 mesh::CellType celltype,
                                 const mesh::GhostMode ghost_mode,
                                 const std::string& diagonal)
{
  return RectangleMesh::create(
      comm, p, n, celltype, ghost_mode,
      static_cast<graph::AdjacencyList<std::int32_t> (*)(
          MPI_Comm, int, int, const graph::AdjacencyList<std::int64_t>&,
          mesh::GhostMode)>(&mesh::partition_cells_graph),
      diagonal);
}
//-----------------------------------------------------------------------------
mesh::Mesh RectangleMesh::create(MPI_Comm comm,
                                 const std::array<std::array<double, 3>, 2>& p,
                                 std::array<std::size_t, 2> n,
                                 mesh::CellType celltype,
                                 const mesh::GhostMode ghost_mode,
                                 const mesh::CellPartitionFunction& partitioner,
                                 const std::string& diagonal)
{
  switch (celltype)
  {
  case mesh::CellType::triangle:
    return build_tri(comm, p, n, ghost_mode, partitioner, diagonal);
  case mesh::CellType::quadrilateral:
    return build_quad(comm, p, n, ghost_mode, partitioner);
  default:
    throw std::runtime_error("Generate rectangle mesh. Wrong cell type");
  }
}
//-----------------------------------------------------------------------------
