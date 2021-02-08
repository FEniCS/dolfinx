// Copyright (C) 2007 Kristian B. Oelgaard
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IntervalMesh.h"
#include <Eigen/Core>
#include <cfloat>
#include <cmath>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>

using namespace dolfinx;
using namespace dolfinx::generation;

namespace
{
mesh::Mesh build(MPI_Comm comm, std::size_t nx, std::array<double, 2> x,
                 const fem::CoordinateElement& element,
                 const mesh::GhostMode ghost_mode,
                 const mesh::CellPartitionFunction& partitioner)
{
  // Receive mesh according to parallel policy
  if (dolfinx::MPI::rank(comm) != 0)
  {
    common::array2d<double> geom(0, 1);
    Eigen::Array<std::int64_t, 0, 2, Eigen::RowMajor> topo(0, 2);
    auto [data, offset] = graph::create_adjacency_data(topo);
    return mesh::create_mesh(
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
  common::array2d<double> geom((nx + 1), 1);
  for (std::size_t ix = 0; ix <= nx; ix++)
    geom(ix, 0) = a + ab * static_cast<double>(ix);

  // Create intervals
  Eigen::Array<std::int64_t, Eigen::Dynamic, 2, Eigen::RowMajor> topo(nx, 2);
  for (std::size_t ix = 0; ix < nx; ix++)
    topo.row(ix) << ix, ix + 1;

  auto [data, offset] = graph::create_adjacency_data(topo);
  return mesh::create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
} // namespace

//-----------------------------------------------------------------------------
mesh::Mesh IntervalMesh::create(MPI_Comm comm, std::size_t n,
                                std::array<double, 2> x,
                                const fem::CoordinateElement& element,
                                const mesh::GhostMode ghost_mode,
                                const mesh::CellPartitionFunction& partitioner)
{
  return build(comm, n, x, element, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------
