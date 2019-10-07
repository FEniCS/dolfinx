// Copyright (C) 2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitTetrahedronMesh.h"
#include <Eigen/Dense>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Partitioning.h>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh UnitTetrahedronMesh::create()
{
  // Create vertices
  Eigen::Array<double, 4, 3, Eigen::RowMajor> geom(4, 3);
  geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  // Create cell
  Eigen::Array<std::int64_t, 1, 4, Eigen::RowMajor> topo(1, 4);
  topo << 0, 1, 2, 3;

  return mesh::Mesh(MPI_COMM_SELF, mesh::CellType::tetrahedron, geom, topo, {},
                    mesh::GhostMode::none);
}
//-----------------------------------------------------------------------------
