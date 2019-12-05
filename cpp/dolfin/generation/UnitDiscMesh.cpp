// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitDiscMesh.h"
#include <cmath>
#include <dolfin/common/types.h>
#include <dolfin/io/cells.h>
#include <dolfin/mesh/Partitioning.h>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh UnitDiscMesh::create(MPI_Comm comm, int n,
                                const mesh::GhostMode ghost_mode)
{
  assert(n > 0);

  // Receive mesh if not rank 0
  if (dolfin::MPI::rank(comm) != 0)
  {
    Eigen::Array<double, 0, 2, Eigen::RowMajor> geom(0, 2);
    Eigen::Array<std::int64_t, 0, 6, Eigen::RowMajor> topo(0, 6);
    return mesh::Partitioning::build_distributed_mesh(
        comm, mesh::CellType::triangle, geom, topo, {}, ghost_mode);
  }

  Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor> points(
      1 + 3 * (2 * n + 1) * 2 * n, 2);
  points.row(0) = 0.0;
  std::uint32_t c = 1;
  for (int i = 1; i < (2 * n + 1); ++i)
  {
    for (int j = 0; j < 6 * i; ++j)
    {
      const double r = (double)i / (double)(2 * n);
      const double th = 2 * M_PI * (double)j / (double)(6 * i);
      const double x = r * cos(th);
      const double y = r * sin(th);
      points.row(c) << x, y;
      ++c;
    }
  }

  // Fill in central circle manually
  Eigen::Array<std::int64_t, Eigen::Dynamic, 6, Eigen::RowMajor> cells(
      6 * n * n, 6);
  cells.block(0, 0, 6, 6) << 0, 7, 9, 1, 8, 2, 0, 9, 11, 2, 10, 3, 0, 11, 13, 3,
      12, 4, 0, 13, 15, 4, 14, 5, 0, 15, 17, 5, 16, 6, 0, 17, 7, 6, 18, 1;
  c = 6;
  for (int i = 0; i < (n - 1); ++i)
  {
    std::uint32_t i1 = 3 * (2 * i + 1) * (2 * i + 2) + 1;
    std::uint32_t i0 = i1;
    std::uint32_t i2 = 3 * (2 * i + 2) * (2 * i + 3) + 1;
    std::uint32_t i3 = 3 * (2 * i + 3) * (2 * i + 4) + 1;
    std::uint32_t i4 = 3 * (2 * i + 4) * (2 * i + 5) + 1;
    for (int k = 0; k < 6; ++k)
    {
      for (int j = 0; j < (i + 1); ++j)
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

  const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      cells_reordered = io::cells::permute_ordering(
          cells,
          io::cells::vtk_to_dolfin(mesh::CellType::triangle, cells.cols()));

  return mesh::Partitioning::build_distributed_mesh(
      comm, mesh::CellType::triangle, points, cells_reordered, {}, ghost_mode);
}

//-----------------------------------------------------------------------------
