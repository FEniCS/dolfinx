// Copyright (C) 2007 Kristian B. Oelgaard
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IntervalMesh.h"
#include "dolfin/common/MPI.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/MeshPartitioning.h"
#include <Eigen/Dense>
#include <cfloat>
#include <cmath>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh IntervalMesh::build(MPI_Comm comm, std::size_t nx,
                               std::array<double, 2> x,
                               const mesh::GhostMode ghost_mode)
{
  // Receive mesh according to parallel policy
  if (MPI::rank(comm) != 0)
  {
    EigenRowArrayXXd geom(0, 1);
    EigenRowArrayXXi64 topo(0, 2);
    return mesh::MeshPartitioning::build_distributed_mesh(
        comm, mesh::CellType::Type::interval, geom, topo, {}, ghost_mode);
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

  EigenRowArrayXXd geom((nx + 1), 1);
  EigenRowArrayXXi64 topo(nx, 2);

  // Create vertices
  for (std::size_t ix = 0; ix <= nx; ix++)
    geom(ix, 0) = a + ab * static_cast<double>(ix);

  // Create intervals
  for (std::size_t ix = 0; ix < nx; ix++)
    topo.row(ix) << ix, ix + 1;

  return mesh::MeshPartitioning::build_distributed_mesh(
      comm, mesh::CellType::Type::interval, geom, topo, {}, ghost_mode);
}
//-----------------------------------------------------------------------------
