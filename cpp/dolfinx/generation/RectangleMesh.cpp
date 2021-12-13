// Copyright (C) 2005-2015 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "RectangleMesh.h"
#include <cfloat>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::generation;

namespace
{
//-----------------------------------------------------------------------------
mesh::Mesh build_tri(MPI_Comm comm,
                     const std::array<std::array<double, 3>, 2>& p,
                     std::array<std::size_t, 2> n, mesh::GhostMode ghost_mode,
                     const mesh::CellPartitionFunction& partitioner,
                     DiagonalType diagonal)
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

  if (nx < 1 or ny < 1)
  {
    throw std::runtime_error(
        "Rectangle has non-positive number of vertices in some dimension: "
        "number of vertices must be at least 1 in each dimension");
  }

  // Create vertices and cells
  std::size_t nv, nc;
  switch (diagonal)
  {
  case DiagonalType::crossed:
    nv = (nx + 1) * (ny + 1) + nx * ny;
    nc = 4 * nx * ny;
    break;
  default:
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
  switch (diagonal)
  {
  case DiagonalType::crossed:
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
    break;
  default:
    break;
  }

  // Create triangles
  switch (diagonal)
  {
  case DiagonalType::crossed:
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
    break;
  }
  default:
  {
    DiagonalType local_diagonal = diagonal;
    for (std::size_t iy = 0; iy < ny; iy++)
    {
      // Set up alternating diagonal
      switch (diagonal)
      {
      case DiagonalType::right_left:
        if (iy % 2)
          local_diagonal = DiagonalType::right;
        else
          local_diagonal = DiagonalType::left;
        break;
      case DiagonalType::left_right:
        if (iy % 2)
          local_diagonal = DiagonalType::left;
        else
          local_diagonal = DiagonalType::right;
        break;
      default:
        break;
      }
      for (std::size_t ix = 0; ix < nx; ix++)
      {
        const std::size_t v0 = iy * (nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);

        std::size_t offset = iy * nx + ix;
        switch (local_diagonal)
        {
        case DiagonalType::left:
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

          switch (diagonal)
          {
          case DiagonalType::right_left: DiagonalType::left_right:
            local_diagonal = DiagonalType::right;
            break;
          default:
            break;
          }
          break;
        }
        default:
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
          switch (diagonal)
          {
          case DiagonalType::right_left: DiagonalType::left_right:
            local_diagonal = DiagonalType::left;
            break;
          default:
            break;
          }
        }
        }
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

//-----------------------------------------------------------------------------
mesh::Mesh build_quad(MPI_Comm comm,
                      const std::array<std::array<double, 3>, 2> p,
                      std::array<std::size_t, 2> n, mesh::GhostMode ghost_mode,
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
} // namespace
//-----------------------------------------------------------------------------
mesh::Mesh RectangleMesh::create(MPI_Comm comm,
                                 const std::array<std::array<double, 3>, 2>& p,
                                 std::array<std::size_t, 2> n,
                                 mesh::CellType celltype,
                                 mesh::GhostMode ghost_mode,
                                 DiagonalType diagonal)
{
  return RectangleMesh::create(comm, p, n, celltype, ghost_mode,
                               mesh::create_cell_partitioner(), diagonal);
}
//-----------------------------------------------------------------------------
mesh::Mesh RectangleMesh::create(MPI_Comm comm,
                                 const std::array<std::array<double, 3>, 2>& p,
                                 std::array<std::size_t, 2> n,
                                 mesh::CellType celltype,
                                 mesh::GhostMode ghost_mode,
                                 const mesh::CellPartitionFunction& partitioner,
                                 DiagonalType diagonal)
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
