// Copyright (C) 2005-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "BoxMesh.h"
#include <Eigen/Dense>
#include <cmath>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/constants.h>
#include <dolfin/mesh/MeshPartitioning.h>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh BoxMesh::create(MPI_Comm comm,
                           const std::array<geometry::Point, 2>& p,
                           std::array<std::size_t, 3> n,
                           mesh::CellType::Type cell_type,
                           const mesh::GhostMode ghost_mode)
{
  if (cell_type == mesh::CellType::Type::tetrahedron)
    return build_tet(comm, p, n, ghost_mode);
  else if (cell_type == mesh::CellType::Type::hexahedron)
    return build_hex(comm, n, ghost_mode);
  else
    throw std::runtime_error("Generate rectangle mesh. Wrong cell type");

  // Will never reach this point
  return build_tet(comm, p, n, ghost_mode);
}
//-----------------------------------------------------------------------------
mesh::Mesh BoxMesh::build_tet(MPI_Comm comm,
                              const std::array<geometry::Point, 2>& p,
                              std::array<std::size_t, 3> n,
                              const mesh::GhostMode ghost_mode)
{
  common::Timer timer("Build BoxMesh");

  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    EigenRowArrayXXd geom(0, 3);
    EigenRowArrayXXi64 topo(0, 4);

    return mesh::MeshPartitioning::build_distributed_mesh(
        comm, mesh::CellType::Type::tetrahedron, geom, topo, {}, ghost_mode);
  }

  // Extract data
  const geometry::Point& p0 = p[0];
  const geometry::Point& p1 = p[1];
  std::size_t nx = n[0];
  std::size_t ny = n[1];
  std::size_t nz = n[2];

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

  if (std::abs(x0 - x1) < DOLFIN_EPS || std::abs(y0 - y1) < DOLFIN_EPS
      || std::abs(z0 - z1) < DOLFIN_EPS)
  {
    throw std::runtime_error(
        "Box seems to have zero width, height or depth. Check dimensions");
  }

  if (nx < 1 || ny < 1 || nz < 1)
  {
    throw std::runtime_error(
        "BoxMesh has non-positive number of vertices in some dimension");
  }

  EigenRowArrayXXd geom((nx + 1) * (ny + 1) * (nz + 1), 3);
  EigenRowArrayXXi64 topo(6 * nx * ny * nz, 4);

  std::size_t vertex = 0;
  for (std::size_t iz = 0; iz <= nz; ++iz)
  {
    const double z = e + ef * static_cast<double>(iz);
    for (std::size_t iy = 0; iy <= ny; ++iy)
    {
      const double y = c + cd * static_cast<double>(iy);
      for (std::size_t ix = 0; ix <= nx; ++ix)
      {
        const double x = a + ab * static_cast<double>(ix);
        geom.row(vertex) << x, y, z;
        ++vertex;
      }
    }
  }

  // Create tetrahedra
  std::size_t cell = 0;
  for (std::size_t iz = 0; iz < nz; ++iz)
  {
    for (std::size_t iy = 0; iy < ny; ++iy)
    {
      for (std::size_t ix = 0; ix < nx; ++ix)
      {
        const std::size_t v0 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        const std::size_t v4 = v0 + (nx + 1) * (ny + 1);
        const std::size_t v5 = v1 + (nx + 1) * (ny + 1);
        const std::size_t v6 = v2 + (nx + 1) * (ny + 1);
        const std::size_t v7 = v3 + (nx + 1) * (ny + 1);

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
    }
  }

  return mesh::MeshPartitioning::build_distributed_mesh(
      comm, mesh::CellType::Type::tetrahedron, geom, topo, {}, ghost_mode);
}
//-----------------------------------------------------------------------------
mesh::Mesh BoxMesh::build_hex(MPI_Comm comm, std::array<std::size_t, 3> n,
                              const mesh::GhostMode ghost_mode)
{
  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    EigenRowArrayXXd geom(0, 3);
    EigenRowArrayXXi64 topo(0, 8);

    return mesh::MeshPartitioning::build_distributed_mesh(
        comm, mesh::CellType::Type::hexahedron, geom, topo, {}, ghost_mode);
  }

  const std::size_t nx = n[0];
  const std::size_t ny = n[1];
  const std::size_t nz = n[2];

  EigenRowArrayXXd geom((nx + 1) * (ny + 1) * (nz + 1), 3);
  EigenRowArrayXXi64 topo(nx * ny * nz, 8);

  const double a = 0.0;
  const double b = 1.0;
  const double c = 0.0;
  const double d = 1.0;
  const double e = 0.0;
  const double f = 1.0;

  // Create main vertices:
  std::size_t vertex = 0;
  for (std::size_t iz = 0; iz <= nz; ++iz)
  {
    const double z
        = e + ((static_cast<double>(iz)) * (f - e) / static_cast<double>(nz));
    for (std::size_t iy = 0; iy <= ny; ++iy)
    {
      const double y
          = c + ((static_cast<double>(iy)) * (d - c) / static_cast<double>(ny));
      for (std::size_t ix = 0; ix <= nx; ix++)
      {
        const double x
            = a
              + ((static_cast<double>(ix)) * (b - a) / static_cast<double>(nx));
        geom.row(vertex) << x, y, z;
        ++vertex;
      }
    }
  }

  // Create cuboids
  std::size_t cell = 0;
  for (std::size_t iz = 0; iz < nz; ++iz)
  {
    for (std::size_t iy = 0; iy < ny; ++iy)
    {
      for (std::size_t ix = 0; ix < nx; ++ix)
      {
        const std::size_t v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix;
        const std::size_t v1 = v0 + 1;
        const std::size_t v2 = v0 + (nx + 1);
        const std::size_t v3 = v1 + (nx + 1);
        const std::size_t v4 = v0 + (nx + 1) * (ny + 1);
        const std::size_t v5 = v1 + (nx + 1) * (ny + 1);
        const std::size_t v6 = v2 + (nx + 1) * (ny + 1);
        const std::size_t v7 = v3 + (nx + 1) * (ny + 1);
        topo.row(cell) << v0, v1, v2, v3, v4, v5, v6, v7;
        ++cell;
      }
    }
  }

  return mesh::MeshPartitioning::build_distributed_mesh(
      comm, mesh::CellType::Type::hexahedron, geom, topo, {}, ghost_mode);
}
//-----------------------------------------------------------------------------
