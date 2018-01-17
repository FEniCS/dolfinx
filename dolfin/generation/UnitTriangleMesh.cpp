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

  // Receive mesh according to parallel policy
  if (MPI::is_receiver(mesh.mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(mesh);
    return mesh;
  }

  // Open mesh for editing
  MeshEditor editor;
  editor.open(mesh, CellType::Type::triangle, 2, 2);

  // Create vertices
  editor.init_vertices_global(3, 3);
  Point x;
  x[0] = 0.0;
  x[1] = 0.0;
  editor.add_vertex(0, 0, x);
  x[0] = 1.0;
  x[1] = 0.0;
  editor.add_vertex(1, 1, x);
  x[0] = 0.0;
  x[1] = 1.0;
  editor.add_vertex(2, 2, x);

  // Create cells
  editor.init_cells_global(1, 1);
  std::vector<std::size_t> cell_data(3);
  cell_data[0] = 0;
  cell_data[1] = 1;
  cell_data[2] = 2;
  editor.add_cell(0, 0, cell_data);

  // Close mesh editor
  editor.close();

  // Broadcast mesh according to parallel policy
  if (MPI::is_broadcaster(mesh.mpi_comm()))
  {
    MeshPartitioning::build_distributed_mesh(mesh);
    return mesh;
  }

  return mesh;
}
//-----------------------------------------------------------------------------
