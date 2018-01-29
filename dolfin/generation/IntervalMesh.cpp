// Copyright (C) 2007 Kristian B. Oelgaard
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IntervalMesh.h"
#include "dolfin/common/MPI.h"
#include "dolfin/common/constants.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/MeshPartitioning.h"

#include <Eigen/Dense>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
void IntervalMesh::build(Mesh& mesh, std::size_t nx, std::array<double, 2> x)
{
  // Receive mesh according to parallel policy
  if (MPI::is_receiver(mesh.mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(mesh);
    return;
  }

  const double a = x[0];
  const double b = x[1];
  const double ab = (b - a) / static_cast<double>(nx);

  if (std::abs(a - b) < DOLFIN_EPS)
  {
    dolfin_error(
        "Interval.cpp", "create interval",
        "Length of interval is zero. Consider checking your dimensions");
  }

  if (b < a)
  {
    dolfin_error("Interval.cpp", "create interval",
                 "Length of interval is negative. Consider checking the order "
                 "of your arguments");
  }

  if (nx < 1)
  {
    dolfin_error("Interval.cpp", "create interval",
                 "Number of points on interval is (%d), it must be at least 1",
                 nx);
  }

  mesh.rename("mesh", "Mesh of the interval (a, b)");

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom((nx + 1), 1);
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> topo(nx, 2);

  // Create vertices
  for (std::size_t ix = 0; ix <= nx; ix++)
    geom(ix, 0) = a + ab * static_cast<double>(ix);

  // Create intervals
  for (std::size_t ix = 0; ix < nx; ix++)
    topo.row(ix) << ix, ix + 1;

  mesh.create(CellType::Type::interval, geom, topo);

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster(mesh.mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(mesh);
    return;
  }
}
//-----------------------------------------------------------------------------
