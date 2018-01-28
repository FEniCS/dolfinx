// Copyright (C) 2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitTriangleMesh.h"
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh UnitTriangleMesh::create()
{
  Mesh mesh(MPI_COMM_SELF);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> geom(3, 2);
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> topo(1, 3);

  // Create vertices
  geom << 0.0, 0.0,
          1.0, 0.0,
          0.0, 1.0;

  topo << 0, 1, 2;

  mesh.create(CellType::Type::triangle, geom, topo);

  return mesh;
}
//-----------------------------------------------------------------------------
