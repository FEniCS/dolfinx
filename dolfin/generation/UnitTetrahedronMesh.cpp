// Copyright (C) 2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitTetrahedronMesh.h"
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <Eigen/Dense>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh UnitTetrahedronMesh::create()
{
  Mesh mesh(MPI_COMM_SELF);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom(4, 3);
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> topo(1, 4);

  // Create vertices
  geom << 0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0;

  // Create cell
  topo << 0, 1, 2, 3;

  mesh.create(CellType::Type::tetrahedron, geom, topo);

  return mesh;
}
//-----------------------------------------------------------------------------
