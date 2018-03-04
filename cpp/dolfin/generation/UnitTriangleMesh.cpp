// Copyright (C) 2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UnitTriangleMesh.h"
#include <Eigen/Dense>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/MeshPartitioning.h>

using namespace dolfin;
using namespace dolfin::generation;

//-----------------------------------------------------------------------------
mesh::Mesh UnitTriangleMesh::create()
{
  EigenRowArrayXXd geom(3, 2);
  EigenRowArrayXXi32 topo(1, 3);

  // Create vertices
  geom << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;

  topo << 0, 1, 2;

  return mesh::Mesh(MPI_COMM_SELF, mesh::CellType::Type::triangle, geom, topo);
}
//-----------------------------------------------------------------------------
