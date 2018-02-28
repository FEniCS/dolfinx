// Copyright (C) 2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitTetrahedronMesh.h"
#include <Eigen/Dense>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/MeshPartitioning.h>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh UnitTetrahedronMesh::create()
{
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom(
      4, 3);
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> topo(1,
                                                                           4);

  // Create vertices
  geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;

  // Create cell
  topo << 0, 1, 2, 3;

  return mesh::Mesh(MPI_COMM_SELF, mesh::CellType::Type::tetrahedron, geom,
                    topo);
}
//-----------------------------------------------------------------------------
