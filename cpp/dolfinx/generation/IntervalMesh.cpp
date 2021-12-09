// Copyright (C) 2007 Kristian B. Oelgaard
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IntervalMesh.h"
#include <cfloat>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::generation;

namespace
{
mesh::Mesh build(MPI_Comm comm, std::size_t nx, std::array<double, 2> x,
                 mesh::GhostMode ghost_mode,
                 const mesh::CellPartitionFunction& partitioner)
{
  fem::CoordinateElement element(mesh::CellType::interval, 1);

  // Receive mesh according to parallel policy
  if (dolfinx::MPI::rank(comm) != 0)
  {
    xt::xtensor<double, 2> geom({0, 1});
    xt::xtensor<std::int64_t, 2> cells({0, 2});
    auto [data, offset] = graph::create_adjacency_data(cells);
    return mesh::create_mesh(
        comm,
        graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
        {element}, geom, ghost_mode, partitioner);
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
  return mesh::create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      {element}, geom, ghost_mode, partitioner);
}
} // namespace

//-----------------------------------------------------------------------------
mesh::Mesh IntervalMesh::create(MPI_Comm comm, std::size_t n,
                                std::array<double, 2> x,
                                mesh::GhostMode ghost_mode,
                                const mesh::CellPartitionFunction& partitioner)
{
  return build(comm, n, x, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------
