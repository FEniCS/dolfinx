// Copyright (C) 2005-2019 Anders Logg, Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "BoxMesh.h"
#include <Eigen/Dense>
#include <cfloat>
#include <cmath>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/io/cells.h>

using namespace dolfin;
using namespace dolfin::generation;

namespace
{
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
create_geom(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
            std::array<std::size_t, 3> n)
{
  // Extract data
  const Eigen::Vector3d& p0 = p[0];
  const Eigen::Vector3d& p1 = p[1];
  std::int64_t nx = n[0];
  std::int64_t ny = n[1];
  std::int64_t nz = n[2];

  const std::int64_t n_points = (nx + 1) * (ny + 1) * (nz + 1);
  std::array<std::int64_t, 2> range_p
      = dolfin::MPI::local_range(comm, n_points);

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
      || std::abs(y0 - y1) < 2.0 * DBL_EPSILON
      || std::abs(z0 - z1) < 2.0 * DBL_EPSILON)
  {
    throw std::runtime_error(
        "Box seems to have zero width, height or depth. Check dimensions");
  }

  if (nx < 1 || ny < 1 || nz < 1)
  {
    throw std::runtime_error(
        "BoxMesh has non-positive number of vertices in some dimension");
  }

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> geom(
      range_p[1] - range_p[0], 3);

  const std::int64_t sqxy = (nx + 1) * (ny + 1);
  for (std::int64_t v = range_p[0]; v < range_p[1]; ++v)
  {
    const std::int64_t iz = v / sqxy;
    const std::int64_t p = v % sqxy;
    const std::int64_t iy = p / (nx + 1);
    const std::int64_t ix = p % (nx + 1);
    const double z = e + ef * static_cast<double>(iz);
    const double y = c + cd * static_cast<double>(iy);
    const double x = a + ab * static_cast<double>(ix);
    geom.row(v - range_p[0]) << x, y, z;
  }

  return geom;
}
//-----------------------------------------------------------------------------
mesh::Mesh build_tet(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
                     std::array<std::size_t, 3> n,
                     const mesh::GhostMode ghost_mode,
                     mesh::Partitioner partitioner)
{
  common::Timer timer("Build BoxMesh");

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> geom
      = create_geom(comm, p, n);

  std::int64_t nx = n[0];
  std::int64_t ny = n[1];
  std::int64_t nz = n[2];
  const std::int64_t n_cells = nx * ny * nz;
  std::array<std::int64_t, 2> range_c = dolfin::MPI::local_range(comm, n_cells);
  Eigen::Array<std::int64_t, Eigen::Dynamic, 4, Eigen::RowMajor> topo(
      6 * (range_c[1] - range_c[0]), 4);

  // Create tetrahedra
  std::int64_t cell = 0;
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

    // Note that v0 < v1 < v2 < v3 < vmid.
    topo.row(cell) << v0, v1, v3, v7;
    ++cell;
    topo.row(cell) << v0, v1, v7, v5;
    ++cell;
    topo.row(cell) << v0, v5, v7, v4;
    ++cell;
    topo.row(cell) << v0, v3, v2, v7;
    ++cell;
    topo.row(cell) << v0, v6, v4, v7;
    ++cell;
    topo.row(cell) << v0, v2, v6, v7;
    ++cell;
  }

  return mesh::Partitioning::build_distributed_mesh(
      comm, mesh::CellType::tetrahedron, geom, topo, {}, ghost_mode,
      partitioner);
}
//-----------------------------------------------------------------------------
mesh::Mesh build_hex(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
                     std::array<std::size_t, 3> n,
                     const mesh::GhostMode ghost_mode,
                     mesh::Partitioner partitioner)
{
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> geom
      = create_geom(comm, p, n);

  const std::int64_t nx = n[0];
  const std::int64_t ny = n[1];
  const std::int64_t nz = n[2];
  const std::int64_t n_cells = nx * ny * nz;
  std::array<std::int64_t, 2> range_c = dolfin::MPI::local_range(comm, n_cells);
  Eigen::Array<std::int64_t, Eigen::Dynamic, 8, Eigen::RowMajor> topo(
      range_c[1] - range_c[0], 8);

  // Create cuboids
  std::int64_t cell = 0;
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
    topo.row(cell) << v0, v4, v2, v6, v1, v5, v3, v7;
    ++cell;
  }

  return mesh::Partitioning::build_distributed_mesh(
      comm, mesh::CellType::hexahedron, geom, topo, {}, ghost_mode,
      partitioner);
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
mesh::Mesh
BoxMesh::create(MPI_Comm comm, const std::array<Eigen::Vector3d, 2>& p,
                std::array<std::size_t, 3> n, mesh::CellType cell_type,
                const mesh::GhostMode ghost_mode, mesh::Partitioner partitioner)
{
  if (cell_type == mesh::CellType::tetrahedron)
    return build_tet(comm, p, n, ghost_mode, partitioner);
  else if (cell_type == mesh::CellType::hexahedron)
    return build_hex(comm, p, n, ghost_mode, partitioner);
  else
    throw std::runtime_error("Generate rectangle mesh. Wrong cell type");

  // Will never reach this point
  return build_tet(comm, p, n, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------
