// Copyright (C) 2010 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-10-19
// Last changed: 2014-02-06

#include <dolfin/common/MPI.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshEditor.h>
#include "UnitTetrahedronMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::Mesh UnitTetrahedronMesh::create()
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
  editor.open(mesh, CellType::Type::tetrahedron, 3, 3);

  // Create vertices
  editor.init_vertices_global(4, 4);
  std::vector<double> x(3);
  x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
  editor.add_vertex(0, x);

  x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
  editor.add_vertex(1, x);

  x[0] = 0.0; x[1] = 1.0; x[2] = 0.0;
  editor.add_vertex(2, x);

  x[0] = 0.0; x[1] = 0.0; x[2] = 1.0;
  editor.add_vertex(3, x);

  // Create cells
  editor.init_cells_global(1, 1);
  std::vector<std::size_t> cell_data(4);
  cell_data[0] = 0; cell_data[1] = 1; cell_data[2] = 2; cell_data[3] = 3;
  editor.add_cell(0, cell_data);

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
