// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitDiscMesh.h"
#include <cmath>
#include <dolfin/common/types.h>
#include <dolfin/mesh/MeshPartitioning.h>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh UnitDiscMesh::create(MPI_Comm comm, std::size_t n,
                                const mesh::GhostMode ghost_mode)
{
  assert(n > 0);

  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    EigenRowArrayXXd geom(0, 2);
    EigenRowArrayXXi64 topo(0, 6);
    return mesh::MeshPartitioning::build_distributed_mesh(
        comm, mesh::CellType::Type::triangle, geom, topo, {},
        ghost_mode);
  }

  EigenRowArrayXXd points(1 + 3 * (2 * n + 1) * 2 * n, 2);

  points.row(0) << 0.0, 0.0;

  std::uint32_t c = 1;
  for (std::size_t i = 1; i < (2 * n + 1); ++i)
  {
    for (std::size_t j = 0; j < 6 * i; ++j)
    {
      const double r = (double)i / (double)(2 * n);
      const double th = 2 * M_PI * (double)j / (double)(6 * i);
      const double x = r * cos(th);
      const double y = r * sin(th);
      points.row(c) << x, y;
      ++c;
    }
  }

  EigenRowArrayXXi64 cells(6 * n * n, 6);

  // Fill in central circle manually
  cells.block(0, 0, 6, 6) << 0, 7, 9, 1, 8, 2, 0, 9, 11, 2, 10, 3, 0, 11, 13, 3,
      12, 4, 0, 13, 15, 4, 14, 5, 0, 15, 17, 5, 16, 6, 0, 17, 7, 6, 18, 1;

  c = 6;
  for (unsigned int i = 0; i < (n - 1); ++i)
  {
    std::uint32_t i1 = 3 * (2 * i + 1) * (2 * i + 2) + 1;
    std::uint32_t i0 = i1;
    std::uint32_t i2 = 3 * (2 * i + 2) * (2 * i + 3) + 1;
    std::uint32_t i3 = 3 * (2 * i + 3) * (2 * i + 4) + 1;
    std::uint32_t i4 = 3 * (2 * i + 4) * (2 * i + 5) + 1;

    for (unsigned int k = 0; k < 6; ++k)
    {
      for (unsigned int j = 0; j < (i + 1); ++j)
      {
        if (j == 0)
        {
          if (k == 0)
            cells.row(c) << i1, i4 - 2, i3, i3 - 1, i4 - 1, i2;
          else
            cells.row(c) << i1, i3 - 2, i3, i2 - 1, i3 - 1, i2;
          ++c;
          cells.row(c) << i1, i3, i3 + 2, i2, i3 + 1, i2 + 1;
          ++c;
          ++i2;
          i3 += 2;
        }
        else
        {
          cells.row(c) << i1, i3 - 2, i3, i2 - 1, i3 - 1, i2;
          ++c;
        }

        if (k == 5 and j == i)
          cells.row(c) << i1, i3, i0, i2, i2 + 1, i1 + 1;
        else
          cells.row(c) << i1, i3, i1 + 2, i2, i2 + 1, i1 + 1;
        ++c;

        i3 += 2;
        i2 += 2;
        i1 += 2;
      }
    }
  }

  return mesh::MeshPartitioning::build_distributed_mesh(
      comm, mesh::CellType::Type::triangle, points, cells, {},
      ghost_mode);
}

//-----------------------------------------------------------------------------
