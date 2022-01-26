// Copyright (C) 2005-2019 Anders Logg, Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "generation.h"
#include <cfloat>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
create_geom(MPI_Comm comm, const std::array<std::array<double, 3>, 2>& p,
            std::array<std::size_t, 3> n)
{
  // Extract data
  const std::array<double, 3>& p0 = p[0];
  const std::array<double, 3>& p1 = p[1];
  std::int64_t nx = n[0];
  std::int64_t ny = n[1];
  std::int64_t nz = n[2];

  const std::int64_t n_points = (nx + 1) * (ny + 1) * (nz + 1);
  std::array range_p = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_points, dolfinx::MPI::size(comm));

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0[0], p1[0]);
  const double x1 = std::max(p0[0], p1[0]);
  const double y0 = std::min(p0[1], p1[1]);
  const double y1 = std::max(p0[1], p1[1]);
  const double z0 = std::min(p0[2], p1[2]);
  const double z1 = std::max(p0[2], p1[2]);

  const double a = x0;
  const double b = x1;
  const double ab = (b - a) / static_cast<double>(nx);
  const double c = y0;
  const double d = y1;
  const double cd = (d - c) / static_cast<double>(ny);
  const double e = z0;
  const double f = z1;
  const double ef = (f - e) / static_cast<double>(nz);

  if (std::abs(x0 - x1) < 2.0 * DBL_EPSILON
      or std::abs(y0 - y1) < 2.0 * DBL_EPSILON
      or std::abs(z0 - z1) < 2.0 * DBL_EPSILON)
  {
    throw std::runtime_error(
        "Box seems to have zero width, height or depth. Check dimensions");
  }

  if (nx < 1 || ny < 1 || nz < 1)
  {
    throw std::runtime_error(
        "BoxMesh has non-positive number of vertices in some dimension");
  }

  xt::xtensor<double, 2> geom(
      {static_cast<std::size_t>(range_p[1] - range_p[0]), 3});
  const std::int64_t sqxy = (nx + 1) * (ny + 1);
  std::array<double, 3> point;
  for (std::int64_t v = range_p[0]; v < range_p[1]; ++v)
  {
    const std::int64_t iz = v / sqxy;
    const std::int64_t p = v % sqxy;
    const std::int64_t iy = p / (nx + 1);
    const std::int64_t ix = p % (nx + 1);
    const double z = e + ef * static_cast<double>(iz);
    const double y = c + cd * static_cast<double>(iy);
    const double x = a + ab * static_cast<double>(ix);
    point = {x, y, z};
    for (std::size_t i = 0; i < 3; i++)
      geom(v - range_p[0], i) = point[i];
  }

  return geom;
}
//-----------------------------------------------------------------------------
mesh::Mesh build_tet(MPI_Comm comm,
                     const std::array<std::array<double, 3>, 2>& p,
                     std::array<std::size_t, 3> n, GhostMode ghost_mode,
                     const CellPartitionFunction& partitioner)
{
  common::Timer timer("Build BoxMesh");

  xt::xtensor<double, 2> geom = create_geom(comm, p, n);

  const std::int64_t nx = n[0];
  const std::int64_t ny = n[1];
  const std::int64_t nz = n[2];
  const std::int64_t n_cells = nx * ny * nz;
  std::array range_c = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_cells, dolfinx::MPI::size(comm));
  const std::size_t cell_range = range_c[1] - range_c[0];
  xt::xtensor<std::int64_t, 2> cells({6 * cell_range, 4});

  // Create tetrahedra
  for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
  {
    const int iz = i / (nx * ny);
    const int j = i % (nx * ny);
    const int iy = j / nx;
    const int ix = j % nx;

    const std::int64_t v0 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (nx + 1);
    const std::int64_t v3 = v1 + (nx + 1);
    const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
    const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
    const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
    const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);

    // Note that v0 < v1 < v2 < v3 < vmid
    xt::xtensor_fixed<std::int64_t, xt::xshape<6, 4>> c
        = {{v0, v1, v3, v7}, {v0, v1, v7, v5}, {v0, v5, v7, v4},
           {v0, v3, v2, v7}, {v0, v6, v4, v7}, {v0, v2, v6, v7}};
    std::size_t offset = 6 * (i - range_c[0]);

    // Note: we would like to assign to a view, but this does not work
    // correctly with the Intel icpx compiler
    // xt::view(cells, xt::range(offset, offset + 6), xt::all()) = c;
    auto _cell = xt::view(cells, xt::range(offset, offset + 6), xt::all());
    _cell.assign(c);
  }

  fem::CoordinateElement element(CellType::tetrahedron, 1);
  auto [data, offset] = graph::create_adjacency_data(cells);
  return create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------
mesh::Mesh build_hex(MPI_Comm comm,
                     const std::array<std::array<double, 3>, 2>& p,
                     std::array<std::size_t, 3> n, GhostMode ghost_mode,
                     const CellPartitionFunction& partitioner)
{
  xt::xtensor<double, 2> geom = create_geom(comm, p, n);

  const std::int64_t nx = n[0];
  const std::int64_t ny = n[1];
  const std::int64_t nz = n[2];
  const std::int64_t n_cells = nx * ny * nz;
  std::array range_c = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_cells, dolfinx::MPI::size(comm));
  const std::size_t cell_range = range_c[1] - range_c[0];
  xt::xtensor<std::int64_t, 2> cells({cell_range, 8});

  // Create cuboids
  for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
  {
    const std::int64_t iz = i / (nx * ny);
    const std::int64_t j = i % (nx * ny);
    const std::int64_t iy = j / nx;
    const std::int64_t ix = j % nx;

    const std::int64_t v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (nx + 1);
    const std::int64_t v3 = v1 + (nx + 1);
    const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
    const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
    const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
    const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);

    xt::xtensor_fixed<std::int64_t, xt::xshape<8>> cell
        = {v0, v1, v2, v3, v4, v5, v6, v7};
    // Note: we would like to assign to a view, but this does not work
    // correctly with the Intel icpx compiler
    // _cell = xt::row(cells, i - range_c[0]) = c;
    auto _cell = xt::row(cells, i - range_c[0]);
    _cell.assign(cell);
  }

  fem::CoordinateElement element(CellType::hexahedron, 1);
  auto [data, offset] = graph::create_adjacency_data(cells);
  return create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------
mesh::Mesh build_prism(MPI_Comm comm,
                       const std::array<std::array<double, 3>, 2>& p,
                       std::array<std::size_t, 3> n, GhostMode ghost_mode,
                       const CellPartitionFunction& partitioner)
{
  xt::xtensor<double, 2> geom = create_geom(comm, p, n);

  const std::int64_t nx = n[0];
  const std::int64_t ny = n[1];
  const std::int64_t nz = n[2];
  const std::int64_t n_cells = nx * ny * nz;
  std::array range_c = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_cells, dolfinx::MPI::size(comm));
  const std::size_t cell_range = range_c[1] - range_c[0];
  xt::xtensor<std::int64_t, 2> cells({cell_range * 2, 6});

  // Create cuboids
  for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
  {
    const std::int64_t iz = i / (nx * ny);
    const std::int64_t j = i % (nx * ny);
    const std::int64_t iy = j / nx;
    const std::int64_t ix = j % nx;

    const std::int64_t v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (nx + 1);
    const std::int64_t v3 = v1 + (nx + 1);
    const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
    const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
    const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
    const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);

    xt::xtensor_fixed<std::int64_t, xt::xshape<6>> cell0
        = {v0, v1, v2, v4, v5, v6};
    xt::xtensor_fixed<std::int64_t, xt::xshape<6>> cell1
        = {v1, v2, v3, v5, v6, v7};

    xt::view(cells, (i - range_c[0]) * 2, xt::all()) = cell0;
    xt::view(cells, (i - range_c[0]) * 2 + 1, xt::all()) = cell1;
  }

  fem::CoordinateElement element(CellType::prism, 1);
  auto [data, offset] = graph::create_adjacency_data(cells);
  return create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
mesh::Mesh mesh::create_box(MPI_Comm comm,
                            const std::array<std::array<double, 3>, 2>& p,
                            std::array<std::size_t, 3> n, CellType celltype,
                            GhostMode ghost_mode,
                            const CellPartitionFunction& partitioner)
{
  switch (celltype)
  {
  case CellType::tetrahedron:
    return build_tet(comm, p, n, ghost_mode, partitioner);
  case CellType::hexahedron:
    return build_hex(comm, p, n, ghost_mode, partitioner);
  case CellType::prism:
    return build_prism(comm, p, n, ghost_mode, partitioner);
  default:
    throw std::runtime_error("Generate box mesh. Wrong cell type");
  }
}

namespace
{
mesh::Mesh build(MPI_Comm comm, std::size_t nx, std::array<double, 2> x,
                 GhostMode ghost_mode, const CellPartitionFunction& partitioner)
{
  fem::CoordinateElement element(CellType::interval, 1);

  // Receive mesh according to parallel policy
  if (dolfinx::MPI::rank(comm) != 0)
  {
    xt::xtensor<double, 2> geom({0, 1});
    xt::xtensor<std::int64_t, 2> cells({0, 2});
    auto [data, offset] = graph::create_adjacency_data(cells);
    return create_mesh(
        comm,
        graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
        element, geom, ghost_mode, partitioner);
  }

  const double a = x[0];
  const double b = x[1];
  const double ab = (b - a) / static_cast<double>(nx);

  if (std::abs(a - b) < DBL_EPSILON)
  {
    throw std::runtime_error(
        "Length of interval is zero. Check your dimensions.");
  }

  if (b < a)
  {
    throw std::runtime_error(
        "Interval length is negative. Check order of arguments.");
  }

  if (nx < 1)
    throw std::runtime_error("Number of points on interval must be at least 1");

  // Create vertices
  xt::xtensor<double, 2> geom({(nx + 1), 1});
  for (std::size_t ix = 0; ix <= nx; ix++)
    geom(ix, 0) = a + ab * static_cast<double>(ix);

  // Create intervals
  xt::xtensor<std::int64_t, 2> cells({nx, 2});
  for (std::size_t ix = 0; ix < nx; ++ix)
    for (std::size_t j = 0; j < 2; ++j)
      cells(ix, j) = ix + j;

  auto [data, offset] = graph::create_adjacency_data(cells);
  return create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
} // namespace

//-----------------------------------------------------------------------------
mesh::Mesh mesh::create_interval(MPI_Comm comm, std::size_t n,
                                 std::array<double, 2> x, GhostMode ghost_mode,
                                 const CellPartitionFunction& partitioner)
{
  return build(comm, n, x, ghost_mode, partitioner);
}

namespace
{
//-----------------------------------------------------------------------------
mesh::Mesh build_tri(MPI_Comm comm,
                     const std::array<std::array<double, 2>, 2>& p,
                     std::array<std::size_t, 2> n, GhostMode ghost_mode,
                     const CellPartitionFunction& partitioner,
                     DiagonalType diagonal)
{
  fem::CoordinateElement element(CellType::triangle, 1);

  // Receive mesh if not rank 0
  if (dolfinx::MPI::rank(comm) != 0)
  {
    xt::xtensor<double, 2> geom({0, 2});
    xt::xtensor<std::int64_t, 2> cells({0, 3});
    auto [data, offset] = graph::create_adjacency_data(cells);
    return create_mesh(
        comm,
        graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
        element, geom, ghost_mode, partitioner);
  }

  const std::array<double, 2> p0 = p[0];
  const std::array<double, 2> p1 = p[1];

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

          if (diagonal == DiagonalType::right_left
              or diagonal == DiagonalType::left_right)
          {
            local_diagonal = DiagonalType::right;
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
          if (diagonal == DiagonalType::right_left
              or diagonal == DiagonalType::left_right)
          {
            local_diagonal = DiagonalType::left;
          }
        }
        }
      }
    }
  }
  }

  auto [data, offset] = graph::create_adjacency_data(cells);
  return create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}

//-----------------------------------------------------------------------------
mesh::Mesh build_quad(MPI_Comm comm,
                      const std::array<std::array<double, 2>, 2> p,
                      std::array<std::size_t, 2> n, GhostMode ghost_mode,
                      const CellPartitionFunction& partitioner)
{
  fem::CoordinateElement element(CellType::quadrilateral, 1);

  // Receive mesh if not rank 0
  if (dolfinx::MPI::rank(comm) != 0)
  {
    xt::xtensor<double, 2> geom({0, 2});
    xt::xtensor<std::int64_t, 2> cells({0, 4});
    auto [data, offset] = graph::create_adjacency_data(cells);
    return create_mesh(
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
  return create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
} // namespace

//-----------------------------------------------------------------------------
mesh::Mesh mesh::create_rectangle(MPI_Comm comm,
                                  const std::array<std::array<double, 2>, 2>& p,
                                  std::array<std::size_t, 2> n,
                                  CellType celltype, GhostMode ghost_mode,
                                  DiagonalType diagonal)
{
  return create_rectangle(comm, p, n, celltype, ghost_mode,
                          create_cell_partitioner(), diagonal);
}
//-----------------------------------------------------------------------------
mesh::Mesh mesh::create_rectangle(MPI_Comm comm,
                                  const std::array<std::array<double, 2>, 2>& p,
                                  std::array<std::size_t, 2> n,
                                  CellType celltype, GhostMode ghost_mode,
                                  const CellPartitionFunction& partitioner,
                                  DiagonalType diagonal)
{
  switch (celltype)
  {
  case CellType::triangle:
    return build_tri(comm, p, n, ghost_mode, partitioner, diagonal);
  case CellType::quadrilateral:
    return build_quad(comm, p, n, ghost_mode, partitioner);
  default:
    throw std::runtime_error("Generate rectangle mesh. Wrong cell type");
  }
}
//-----------------------------------------------------------------------------
