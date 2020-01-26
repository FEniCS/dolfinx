// Copyright (C) 2010 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitTriangleMesh.h"
#include <Eigen/Dense>
#include <dolfinx/common/MPI.h>
#include <dolfinx/mesh/Partitioning.h>

using namespace dolfinx;
using namespace dolfinx::generation;

//-----------------------------------------------------------------------------
mesh::Mesh UnitTriangleMesh::create()
{

  // Create vertices
  Eigen::Array<double, 3, 2, Eigen::RowMajor> geom(3, 2);
  geom << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;

  // Create cell
  Eigen::Array<std::int64_t, 1, 3, Eigen::RowMajor> topo(1, 3);
  topo << 0, 1, 2;

  return mesh::Mesh(MPI_COMM_SELF, mesh::CellType::triangle, geom, topo, {},
                    mesh::GhostMode::none);
}
//-----------------------------------------------------------------------------
